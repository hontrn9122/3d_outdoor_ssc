import os
import torch
import numpy as np
import yaml
import math
from model_spconv.layers import *
from model_spconv.networks import *
import lightning as L
import torch.nn.functional as F
from model_spconv.utils import *
from model_spconv.dataloader import SSCDataset
from spconv.pytorch import SparseConvTensor
from .cylinder_fea_generator import *
from .segmentator_3d_asymm_spconv import *
from .unet_2D import *

class SSCNetwork(L.LightningModule):
    def __init__(
        self,
        in_channels=10,
        num_classes=20,
        point_neighbors=16,
        fus_mode='concat',
        data_config_path ='model_spconv/cfgs/semantic-kitti.yaml',
        **kwargs
    ):
        super(SSCNetwork, self).__init__()
        self.data_config = yaml.safe_load(open(data_config_path, 'r'))
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.cfeat_extractor = cylinder_fea(in_channels, out_channels=256, compr_channels=16)
        self.ss_net = Asymm_3d_spconv(num_input_features=16, nclasses=20, init_size=32)
        self.pc_refinement = PCRefinement()
        self.sc_net = BEV_UNet(
            n_class=640,
            dilation=1,
            bilinear=True,
            group_conv=False,
            input_batch_norm=True,
            dropout=0.5,
            circular_padding=False,
            dropblock=False,
        )

        self.w_margin = 1
        self.w_sem = [1, 2, 1, 2]
        self.w_com = [2, 4]
        # self.w_lovasz = [1, 2, 1, 2]
        # self.w_wce = [1, 0.5, 0.5, 0.5]


        learning_map = self.data_config["learning_map"]
        learning_map_inv = self.data_config["learning_map_inv"]
        id_to_classes = self.data_config["labels"]
        self.class_names = {
          k: id_to_classes[learning_map_inv[k]]
          for k in learning_map_inv
        }

        # class weight calculation
        # epsilon_w = 0.001  # eps to avoid zero division
        # sc_class_freq = self.data_config["sc_class_freq"]
        # ss_class_freq = self.data_config["ss_class_freq"]
        # self.sc_class_weights = torch.from_numpy(1 / np.log(np.array(sc_class_freq) + epsilon_w)).float()
        # self.ss_class_weights = torch.from_numpy(1 / np.log(np.array(ss_class_freq) + epsilon_w)).float()

        class_freq = self.data_config["content"]
        self.class_freq = torch.zeros(self.num_classes)
        for i, c in class_freq.items():
            self.class_freq[learning_map[i]] = self.class_freq[learning_map[i]] + c
        self.sc_class_weights = torch.log(1 / self.class_freq)

        self.set_optimizers()

        self.evaluator = iouEval(num_classes, [])
        
        self.sem_evaluator = iouEval(num_classes, [0])

        self.train_stage = 'all'
        
    def reset_time(self):
        self.inf_time = 0
        self.count = 0
        
    def update_time(self, time):
        self.inf_time += time
        self.count += 1
        
    def cal_avg_time(self):
        return self.inf_time/815

    def freeze_stage1_param(self):
        for param in self.cfeat_extractor.parameters():
            param.requires_grad = False
        for param in self.ss_net.parameters():
            param.requires_grad = False
    
    def unfreeze_stage1_param(self):
        for param in self.cfeat_extractor.parameters():
            param.requires_grad = True
        for param in self.ss_net.parameters():
            param.requires_grad = True
    
    def set_optimizers(self, optim=None):
        if optim == None:
            optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.optim = optim

    def set_schedulers(self, scheduler=None):
        self.scheduler = scheduler

    def configure_optimizers(self):
        return {
            "optimizer": self.optim,
            "lr_scheduler": self.scheduler,
        }

    def set_train_stage(self, stage):
        assert stage in [1, 2, 'all']
        self.train_stage = stage


    def forward_trainval_stage1(self, point, cylinder, batch_size, qinfor):
        # forward pass
        cylinder, pfeat = self.cfeat_extractor(point, cylinder, batch_size, qinfor, stage=1)
        y, py, _ = self.ss_net(cylinder, pfeat, point.p2v)
        return y, py

    def forward_trainval_stage2(self, point, cylinder, voxel, batch_size, qinfor):
        cylinder, pfeat, evoxel = self.cfeat_extractor(point, cylinder, batch_size, qinfor, stage=2)
        feats = self.ss_net.encode(cylinder, pfeat, point, batch_size)
        
        feats = self.pc_refinement(pfeat, point, feats, batch_size, qinfor)
        
        evoxel = torch.max(evoxel, dim=-1, keepdim=False).values
        pred, caps = self.sc_net(voxel, evoxel, *feats)
        pred = pred.permute(0,2,3,1)   # [B, 256, 256, 640]
        new_shape = list(pred.shape)[:3] + [32, 20]    # [B, 256, 256, 32, 20]
        pred = pred.view(new_shape)    # [B, 256, 256, 32, 20]
        pred = pred.permute(0,4,1,2,3)   # [B, 20, 256, 32, 256]  # [B, C, H, W, D] -> [B, C, W, H, D]
        return pred, caps
    
    def forward_trainval_all(self, point, cylinder, voxel, batch_size, qinfor):
        # forward pass
        cylinder, pfeat, evoxel = self.cfeat_extractor(point, cylinder, batch_size, qinfor, stage=2)
        y, py, feats = self.ss_net(cylinder, pfeat, point.p2v)
        
        feats = self.pc_refinement(pfeat, point, feats, batch_size, qinfor)
        
        evoxel = torch.max(evoxel, dim=-1, keepdim=False).values
        pred, caps = self.sc_net(voxel, evoxel, *feats)
        pred = pred.permute(0,2,3,1)   # [B, 256, 256, 640]
        new_shape = list(pred.shape)[:3] + [32, 20]    # [B, 256, 256, 32, 20]
        pred = pred.view(new_shape)    # [B, 256, 256, 32, 20]
        pred = pred.permute(0,4,1,2,3)   # [B, 20, 256, 32, 256]  # [B, C, H, W, D] -> [B, C, W, H, D]
        
        return y, py, pred, caps
    
    def forward(self, data_batch, save=True, save_path='predictions/sequences/', inv_remap_lut=None, device='cpu'):
        file_info, collection, batch_size, qinfor = data_batch
        collection = {k: v.to(device) for k, v in collection.items()}
        qinfor = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in qinfor.items()}
        point = collection['point']
        cylinder = collection['cylinder_coord']
        voxel = collection['input'].permute(0, 3, 1, 2)
        
        start_time = time.time()
        
        cylinder, pfeat, evoxel = self.cfeat_extractor(point, cylinder, batch_size, qinfor, stage=2)
        feats = self.ss_net.encode(cylinder, pfeat, point, batch_size)
        
        feats = self.pc_refinement(pfeat, point, feats, batch_size, qinfor)
        
        evoxel = torch.max(evoxel, dim=-1, keepdim=False).values
        pred, caps = self.sc_net(voxel, evoxel, *feats)
        inf_time = time.time() - start_time
        self.update_time(inf_time)
        
        pred = pred.permute(0,2,3,1)   # [B, 256, 256, 640]
        new_shape = list(pred.shape)[:3] + [32, 20]    # [B, 256, 256, 32, 20]
        pred = pred.view(new_shape)    # [B, 256, 256, 32, 20]
        pred = pred.permute(0,4,1,2,3)   # [B, 20, 256, 32, 256]  # [B, C, H, W, D] -> [B, C, W, H, D]
        
        pred = torch.argmax(pred, dim=1)
        if save:
            scores = pred.cpu().numpy()
            for i in range(len(file_info)):
                score = scores[i].reshape(-1).astype(np.uint16)
                score = inv_remap_lut[score].astype(np.uint16)
                out_dir = os.path.join(save_path, file_info[i][0], 'predictions')
                os.makedirs(out_dir, exist_ok=True)
                out_filename = os.path.join(out_dir, f'{file_info[i][1]}.label')
                score.tofile(out_filename)
        return pred

    def training_step(self, train_batch, batch_idx):
        file_info, collection, batch_size, qinfor = train_batch
        point = collection['point']
        cylinder_coord = collection['cylinder_coord']
        
        labels = []
        if self.train_stage != 2:
            labels.extend([collection['cylinder_label'], point.y])
            invalids = None
        if self.train_stage != 1:
            voxel = collection['input'].permute(0, 3, 1, 2)
            labels.extend([collection['label']])
            invalids = [collection['invalid']]

        if self.train_stage == 1:
            # forward pass
            res = self.forward_trainval_stage1(point, cylinder_coord, batch_size, qinfor)

        elif self.train_stage == 2:
            res = self.forward_trainval_stage2(point, cylinder_coord, voxel, batch_size, qinfor)
            
        else:
            res = self.forward_trainval_all(point, cylinder_coord, voxel, batch_size, qinfor)

        # losses calculation
        losses, total_loss = self.losses_cal(res, labels, invalids)
        losses = {'train_' + k: losses[k] for k in losses.keys()}

        # log results
        self.log_dict(losses, batch_size=batch_size, sync_dist=True)

        return total_loss


    def validation_step(self, val_batch, batch_idx):
        file_info, collection, batch_size, qinfor = val_batch
        point = collection['point']
        cylinder_coord = collection['cylinder_coord']
        
        labels = []
        if self.train_stage != 2:
            labels.extend([collection['cylinder_label'], point.y])
            invalids = None
        if self.train_stage != 1:
            voxel = collection['input'].permute(0, 3, 1, 2)
            labels.extend([collection['label']])
            invalids = [collection['invalid']]

        if self.train_stage == 1:
            # forward pass
            res = self.forward_trainval_stage1(point, cylinder_coord, batch_size, qinfor)
            predicted = one_hot_reverse(F.softmax(res[1], dim=1), dim=1)
            self.sem_eval_metrics_update(predicted, point.y)
        elif self.train_stage == 2:
            res = self.forward_trainval_stage2(point, cylinder_coord, voxel, batch_size, qinfor)
            # eval metrics calculation
            predicted = one_hot_reverse(F.softmax(res[0], dim=1), dim=1)
            self.eval_metrics_update(predicted, labels[0], invalids[0])
        else:
            res = self.forward_trainval_all(point, cylinder_coord, voxel, batch_size, qinfor)
            # eval metrics calculation
            predicted = one_hot_reverse(F.softmax(res[2], dim=1), dim=1)
            self.eval_metrics_update(predicted, labels[2], invalids[0])

        # losses calculation
        losses, total_loss = self.losses_cal(res, labels, invalids)
        losses = {'val_' + k: losses[k] for k in losses.keys()}

        self.log_dict(losses, batch_size=batch_size, sync_dist=True)


    def losses_cal(self, res, labels, invalids):
        losses = {}
        device = res[0].device
        sc_class_weights = self.sc_class_weights.to(device)
        # ss_class_weights = self.ss_class_weights.to(device)

        if self.train_stage != 2:
            # losses['l_focal_point'] = self.w_sem[0] * F.cross_entropy(res[1], point_label[0], weight=sc_class_weights)
            losses['l_focal_point'] = self.w_sem[0] * FocalLoss(gamma=3, alpha=sc_class_weights)(res[1], labels[1])
            losses['l_lovasz_point'] = self.w_sem[1] * lovasz_softmax(F.softmax(res[1], dim=1), labels[1], ignore=0)
            
            losses['l_focal_cy'] = self.w_sem[2] * FocalLoss(gamma=3, alpha=sc_class_weights)(res[0], labels[0])
            # losses['l_focal_cy'] = self.w_sem[2] * F.cross_entropy(res[0], point_label[1], weight=sc_class_weights)
            losses['l_lovasz_cy'] = self.w_sem[3] * lovasz_softmax(F.softmax(res[0], dim=1), labels[0], ignore=0)

        if self.train_stage != 1:
            if self.train_stage == 'all':
                res = res[2:]
                labels = labels[2:]
                
            labels[0][invalids[0]==1] = 255
            
            # calculate soft downsample label for capsule margin loss
            with torch.no_grad():
                caps_invalid = torch.where(labels[0] == 255, True, False)
                caps_invalid = batch_bev_majority_pooling(
                    caps_invalid.unsqueeze(1),
                    kernel_size=16
                ).squeeze(1) # shape (B, H//8, W//8)

                caps_label = torch.where(labels[0] == 255, 0, labels[0])
                caps_label = F.softmax(nn.AvgPool2d(16, stride=16, divisor_override=1)(
                    torch.sum(
                        one_hot(caps_label, 20, dim=4).permute(0,4,1,2,3), # shape (B, N, H, W, D)
                        dim=4, keepdim=False
                    ) # shape (B, N, H, W)
                ).permute(0,2,3,1), dim=3)  #/math.log2(8192)).contiguous() # shape (B, H//8, W//8, N)

            caps_norm = torch.linalg.norm(res[1], dim=2)
            
            # l_margin to control CapsNet
            losses['l_margin'] = self.w_margin * MarginLoss(class_weight=sc_class_weights, margin=0.2)(caps_norm, caps_label, caps_invalid)

            losses['l_focal'] = self.w_com[0] * F.cross_entropy(res[0], labels[0], weight=sc_class_weights, ignore_index=255)
            losses['l_lovasz'] = self.w_com[1] * lovasz_softmax(F.softmax(res[0], dim=1), labels[0], ignore=255)
            # l_margin = 0

        losses['total_loss'] = total_losses = sum(losses.values())
        losses = {k: v.item() for k, v in losses.items()}
        return losses, total_losses

    def get_eval_mask(self, label, invalid):
        mask = torch.ones_like(label, dtype=torch.bool)
        mask[label == 255] = False
        mask[invalid == 1] = False

        return mask

    def eval_metrics_update(self, pred, label, invalid):
#         labels, invalid = targets
#         label = labels[0]

        mask = self.get_eval_mask(label, invalid)

        label = label[mask]
        pred = pred[mask]

        self.evaluator.addBatch(pred, label)

    def sem_eval_metrics_update(self, pred, label):
        self.sem_evaluator.addBatch(pred, label)


    def eval_metrics_log(self):
        ignore = [0]
        if self.train_stage == 1:
            m_accuracy = self.sem_evaluator.getacc()
            m_jaccard, class_jaccard = self.sem_evaluator.getIoU()
            metrics = {}
            metrics['accuracy'] = m_accuracy
            metrics['iou_mean'] = m_jaccard

            for i, jacc in enumerate(class_jaccard):
                if i not in ignore:
                    metrics["iou_" + self.class_names[i]] = float(jacc)
            self.log_dict(metrics, sync_dist=True)
            self.sem_evaluator.reset()
        else:
            _, class_jaccard = self.evaluator.getIoU()
            m_jaccard = class_jaccard[1:].mean()
            import numpy as np
            epsilon = np.finfo(np.float32).eps

            # compute remaining metrics.
            conf = self.evaluator.get_confusion()
            precision = torch.sum(conf[1:,1:]) / (torch.sum(conf[1:,:]) + epsilon)
            recall = torch.sum(conf[1:,1:]) / (torch.sum(conf[:,1:]) + epsilon)
            acc_cmpltn = (torch.sum(conf[1:, 1:])) / (torch.sum(conf) - conf[0,0])
            mIoU_ssc = m_jaccard

            metrics = {}
            metrics["iou_completion"] = float(acc_cmpltn)
            metrics["iou_mean"] = float(mIoU_ssc)
            metrics["recall"] = float(recall)
            metrics["precision"] = float(precision)

            for i, jacc in enumerate(class_jaccard):
                if i not in ignore:
                    metrics["iou_" + self.class_names[i]] = float(jacc)

            self.log_dict(metrics, sync_dist=True)
            self.evaluator.reset()


class SSCNetworkCaps(SSCNetwork):
    def __init__(
        self,
        in_channels=10,
        num_classes=20,
        point_neighbors=16,
        fus_mode='concat',
        data_config_path ='model_spconv/cfgs/semantic-kitti.yaml',
        **kwargs
    ):
        L.LightningModule.__init__(self)
        self.data_config = yaml.safe_load(open(data_config_path, 'r'))
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.cfeat_extractor = cylinder_fea(in_channels, out_channels=256, compr_channels=16)
        self.ss_net = Asymm_3d_spconv(num_input_features=16, nclasses=20, init_size=32)
        self.pc_refinement = PCRefinement()
        self.sc_net = CapsBEV_UNet(
            n_class=640,
            dilation=1,
            circular_padding=False,
        )

        self.w_margin = 1
        self.w_sem = [1, 2, 1, 2]
        self.w_com = [2, 4]
        # self.w_lovasz = [1, 2, 1, 2]
        # self.w_wce = [1, 0.5, 0.5, 0.5]


        learning_map = self.data_config["learning_map"]
        learning_map_inv = self.data_config["learning_map_inv"]
        id_to_classes = self.data_config["labels"]
        self.class_names = {
          k: id_to_classes[learning_map_inv[k]]
          for k in learning_map_inv
        }

        # class weight calculation
        # epsilon_w = 0.001  # eps to avoid zero division
        # sc_class_freq = self.data_config["sc_class_freq"]
        # ss_class_freq = self.data_config["ss_class_freq"]
        # self.sc_class_weights = torch.from_numpy(1 / np.log(np.array(sc_class_freq) + epsilon_w)).float()
        # self.ss_class_weights = torch.from_numpy(1 / np.log(np.array(ss_class_freq) + epsilon_w)).float()

        class_freq = self.data_config["content"]
        self.class_freq = torch.zeros(self.num_classes)
        for i, c in class_freq.items():
            self.class_freq[learning_map[i]] = self.class_freq[learning_map[i]] + c
        self.sc_class_weights = torch.log(1 / self.class_freq)

        self.set_optimizers()

        self.evaluator = iouEval(num_classes, [])
        
        self.sem_evaluator = iouEval(num_classes, [0])

        self.train_stage = 'all'
        
    def forward_trainval_stage2(self, point, cylinder, voxel, batch_size, qinfor):
        cylinder, pfeat, evoxel = self.cfeat_extractor(point, cylinder, batch_size, qinfor, stage=2)
        feats = self.ss_net.encode(cylinder, pfeat, point, batch_size)
        
        feats = self.pc_refinement(pfeat, point, feats, batch_size, qinfor)
        
        evoxel = evoxel.permute(0, 1, 4, 2, 3)
        evoxel_shape = list(evoxel.shape)
        evoxel_shape = evoxel_shape[0:1] + [-1] + evoxel_shape[3:]
        evoxel = evoxel.reshape(*evoxel_shape)
        pred, caps = self.sc_net(voxel, evoxel, *feats)
        # pred = pred.permute(0,2,3,1)   # [B, 256, 256, 640]
        # new_shape = list(pred.shape)[:3] + [32, 20]    # [B, 256, 256, 32, 20]
        # pred = pred.view(new_shape)    # [B, 256, 256, 32, 20]
        # pred = pred.permute(0,4,1,2,3)   # [B, 20, 256, 32, 256]  # [B, C, H, W, D] -> [B, C, W, H, D]
        return pred, caps
    
    
    def forward(self, data_batch, save=True, save_path='predictions/sequences/', inv_remap_lut=None, device='cpu'):
        file_info, collection, batch_size, qinfor = data_batch
        collection = {k: v.to(device) for k, v in collection.items()}
        qinfor = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in qinfor.items()}
        point = collection['point']
        cylinder = collection['cylinder_coord']
        voxel = collection['input'].permute(0, 3, 1, 2)
        
        start_time = time.time()
        cylinder, pfeat, evoxel = self.cfeat_extractor(point, cylinder, batch_size, qinfor, stage=2)
        feats = self.ss_net.encode(cylinder, pfeat, point, batch_size)
        
        feats = self.pc_refinement(pfeat, point, feats, batch_size, qinfor)
        
        evoxel = evoxel.permute(0, 1, 4, 2, 3)
        evoxel_shape = list(evoxel.shape)
        evoxel_shape = evoxel_shape[0:1] + [-1] + evoxel_shape[3:]
        evoxel = evoxel.reshape(*evoxel_shape)
        pred, caps = self.sc_net(voxel, evoxel, *feats)
        self.update_time(time.time() - start_time)
        
        pred = torch.argmax(pred, dim=1)
        if save:
            scores = pred.cpu().numpy()
            for i in range(len(file_info)):
                score = scores[i].reshape(-1).astype(np.uint16)
                score = inv_remap_lut[score].astype(np.uint16)
                out_dir = os.path.join(save_path, file_info[i][0], 'predictions')
                os.makedirs(out_dir, exist_ok=True)
                out_filename = os.path.join(out_dir, f'{file_info[i][1]}.label')
                score.tofile(out_filename)
        return pred