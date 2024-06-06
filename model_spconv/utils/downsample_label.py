import yaml
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from .one_hot import one_hot
from .majority_pooling import majority_pooling
from .coord_transform import cart2polar
from .quantize import ravel_hash
from model_spconv.dataloader import SSCDataset
from torch_geometric.data import Data
from torch_geometric.utils import scatter
import math

data_config = yaml.safe_load(open('model_spconv/cfgs/semantic-kitti.yaml', 'r'))
SEQUENCES = data_config["split"]['train'] + data_config["split"]['valid']
VOXEL_DIMS = (256, 256, 32)
LEARNING_MAP = data_config["learning_map"]
MIN_EXTENT = torch.tensor([[0, -25.6, -2]])
MAX_EXTENT = torch.tensor([[51.2, 25.6, 4.4]])
CYL_VOXEL_DIMS = torch.tensor([[480, 180, 32]])
CYL_MAX_EXTENT = torch.tensor([math.sqrt(51.2**2 + 25.6**2), math.pi, 4.4])
CYL_MIN_EXTENT = torch.tensor([[0, 0, -2]])

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def downsample_label(data_root, sequences = SEQUENCES, device='cpu'):
    for sequence in tqdm(sequences, desc="Sequence: "):
        sequence = str(sequence).zfill(2)
        complete_path = os.path.join(data_root, "sequences", sequence, "voxels")
        if not os.path.exists(complete_path): raise RuntimeError("Voxel directory missing: " + complete_path)
        
        files = os.listdir(complete_path)
        label_data = sorted([os.path.join(complete_path, f) for f in files if f.endswith('.label')])
        invalid_data = sorted([os.path.join(complete_path, f) for f in files if f.endswith('.invalid')])
        if len(label_data) == 0: raise RuntimeError("Missing data for label!")
        if len(invalid_data) == 0: raise RuntimeError("Missing data for invalid!")
        
        for label_file, invalid_file in tqdm(list(zip(label_data, invalid_data)), desc= "Scan: ", leave=False):
            label = np.fromfile(label_file, dtype=np.uint16)
            invalid = torch.from_numpy(unpack(np.fromfile(invalid_file, dtype=np.uint8)).astype('float32')).to(device)
            invalid = invalid.reshape(VOXEL_DIMS)

            # label for capsule margin loss
            smt_label = SSCDataset.get_remap_lut(LEARNING_MAP, completion=True)[label]
            smt_label = torch.from_numpy(smt_label.astype('int64')).to(device)
            smt_label = smt_label.reshape(VOXEL_DIMS)
            smt_invalid = invalid.clone()
            smt_mask = smt_label==255
            smt_invalid[smt_mask] = 1
            smt_label[smt_mask] = 0
            if os.path.exists(f'{label_file}_8'): os.remove(f'{label_file}_8')
            torch.save(nn.Softmax(dim=0)(
                nn.AvgPool3d(8, stride=8, divisor_override=1)(
                    one_hot(smt_label, 20, dim=3).permute(3,0,1,2),
                )
            ).permute(1,2,3,0).cpu(), f'{label_file}_8')
            
            #invalid for margin loss label
            torch.save(majority_pooling(
                smt_invalid.unsqueeze(0), 
                kernel_size=8
            ).squeeze(0).cpu(), f'{invalid_file}_{8}')
                
            
            # label for scaled completion
            ssc_label = SSCDataset.get_remap_lut(LEARNING_MAP, completion=True)[label]
            ssc_label = torch.from_numpy(ssc_label.astype('int64')).to(device)
            ssc_label = ssc_label.reshape(VOXEL_DIMS)
            ssc_label[invalid==1] = 255
            for d in [2, 4]:
                if os.path.exists(f'{label_file}_{d}'): os.remove(f'{label_file}_{d}')
                torch.save(majority_pooling(
                    ssc_label.unsqueeze(0), 
                    kernel_size=d
                ).squeeze(0).cpu(), f'{label_file}_{d}')
                

def process_point_label(data_root, sequences = SEQUENCES, device='cpu'):
    for sequence in tqdm(sequences, desc="Sequence"):
        sequence = str(sequence).zfill(2)
        complete_path = os.path.join(data_root, "sequences", sequence, "voxels")
        point_path = os.path.join(data_root, "sequences", sequence, "velodyne")
        point_label_path = os.path.join(data_root, "sequences", sequence, "labels")
        if not os.path.exists(complete_path): raise RuntimeError("Voxel directory missing: " + complete_path)
        if not os.path.exists(point_path): raise RuntimeError("Point directory missing: " + point_path)
        if not os.path.exists(point_label_path): raise RuntimeError("Voxel directory missing: " + point_label_path)

        files = os.listdir(complete_path)
        label_data = sorted([os.path.join(point_label_path, f) for f in files if f.endswith('.label')])
        point_data = sorted([os.path.join(point_path, f) for f in files if f.endswith('.bin')])
        cyl_voxel_size = (CYL_MAX_EXTENT - CYL_MIN_EXTENT) / CYL_VOXEL_DIMS
        # print('voxel_size: ', cyl_voxel_size)

        for label_file, point_file in tqdm(list(zip(label_data, point_data)), desc= "Scan: ", leave=False):
#             # get cordinate and remission infor from raw point
#             point = np.fromfile(point_file, dtype=np.float32).reshape((-1, 4))
#             coord = torch.from_numpy(point[:,:3]).to(device)
#             feat = torch.from_numpy(point[:,3:]).to(device)
            
#             # get label data
#             label = np.fromfile(label_file, dtype=np.uint32)
#             label = label.reshape((-1))  & 0xFFFF # remove instance infor
#             label = SSCDataset.get_remap_lut(LEARNING_MAP, completion=False)[label]
#             label = torch.from_numpy(label.astype('int64')).to(device)
            
#             # filter point in pre-defined range
#             mask = torch.all(
#                 ((coord >= MIN_EXTENT.to(device)) & 
#                  (coord <= MAX_EXTENT.to(device))), 
#                 dim=1
#             )
#             coord = coord[mask]
#             feat = feat[mask]
#             label = label[mask]
            
#             # transform to polar coord
#             pcoord = torch.clamp(cart2polar(coord), min=CYL_MIN_EXTENT[0].to(device), max=CYL_MAX_EXTENT[0].to(device))
            
#             # voxelize polar coord (cylinder partition)
#             quantized_pcoord = torch.floor((pcoord - CYL_MIN_EXTENT.to(device))* (1 / cyl_voxel_size.to(device)))
#             # print('quantized_polar_coord_max: ', torch.max(quantized_pcoord, dim=0).values)
# #             hashed_coords = ravel_hash(quantized_pcoord, xmax = CYL_VOXEL_DIMS[0])
            
# #             sorted_qcoords, sorted_idx = torch.sort(hashed_coords, stable=True)
            
# #             coord = coord[sorted_idx]
# #             feat = feat[sorted_idx]
# #             pcoord = pcoord[sorted_idx]
# #             label = label[sorted_idx]
# #             quantized_pcoord = quantized_pcoord[sorted_idx]
            
#             pvcoord, p2v_map, offsets = torch.unique(
#                 quantized_pcoord, sorted=True, 
#                 return_inverse=True, return_counts=True, dim=0
#             )

#             offsets = scatter(torch.ones_like(p2v_map), p2v_map)
#             sizes = offsets.tolist()
#             # print(sorted_qcoords.shape, batch_idx.shape)
#             label_list = torch.split(label, sizes)
            
#             # pooling point label for voxel label
#             vlabel_list = []
#             for l in label_list:
#                 if torch.all(l==0):
#                     vlabel_list.append(torch.zeros_like(l[0]).unsqueeze(0))
#                 else:
#                     vlabel_list.append(l[l!=0].mode(dim=0, keepdim=True).values)
#             vlabel = torch.cat(vlabel_list, dim=0)
#             assert vlabel.shape[0] == pvcoord.shape[0]
#             pvcoord = quantized_pcoord[[0] + sizes[:-1]]
#             feat = torch.cat([pcoord, coord, feat], dim=1)

            
#             point_data = {
#                 "point": Data(
#                     coord = coord,
#                     x = feat,
#                     y = label,
#                     p_coord = pcoord,
#                     p2v = p2v_map,
#                 ).cpu(),
#                 "cylinder": Data(
#                     x = torch.zeros_like(vlabel),
#                     coord = pvcoord,
#                     y = vlabel,
#                 ).cpu()
#             }
            if os.path.exists(f'{point_file[:-4]}.data'): os.remove(f'{point_file[:-4]}.data')
            # torch.save(point_data, f'{point_file[:-4]}.data')

            