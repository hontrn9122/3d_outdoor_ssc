import os
import math
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset
from model_spconv.utils import dense_to_sparse_info, sparse_quantize
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from functools import partial

# import torch_geometric.transforms as T
# from torch_geometric.transforms import SamplePoints, KNNGraph
# import KPConv.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors


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


SPLIT_SEQUENCES = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "valid": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
}

SPLIT_FILES = {
    "train": [".bin", ".label", ".invalid", ".occluded",],
    "valid": [".bin", ".label", ".invalid", ".occluded",],
    "test": [".bin"]
}

EXT_TO_NAME = {".bin": "input", ".label": "label", ".invalid": "invalid", ".occluded": "occluded", }


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.atan2(input_xyz[:, 0], input_xyz[:, 1])
    return torch.stack((rho, phi, input_xyz[:, 2]), dim=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * torch.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * torch.sin(input_xyz_polar[1])
    return torch.stack((x, y, input_xyz_polar[2]), dim=0)


class SSCDataset(Dataset):
    def __init__(
        self, 
        data_root, 
        data_config_path ='model_spconv/cfgs/semantic-kitti.yaml', 
        split="train",
        augmentation=True,
        stage=1
        # shuffle_index=True,
    ):
        """ Load data from given dataset directory. """
        self.stage = stage
        self.voxel_dims = (256, 256, 32)
        self.min_extent = torch.tensor([[0, -25.6, -2]])
        self.max_extent = torch.tensor([[51.2, 25.6, 4.4]])
        if stage == 1:
            self.cyl_voxel_dims = torch.tensor([[480, 360, 32]])
            self.cyl_max_extent = torch.tensor([math.ceil(math.sqrt(51.2**2 + 25.6**2) - 1), math.pi, 4.4])
            self.cyl_min_extent = torch.tensor([[0, -math.pi, -2]])
        else:
            self.cyl_voxel_dims = torch.tensor([[480, 180, 32]])
            self.cyl_max_extent = torch.tensor([math.ceil(math.sqrt(51.2**2 + 25.6**2) - 1), math.pi, 4.4])
            self.cyl_min_extent = torch.tensor([[0, 0, -2]])

        self.split = split
        self.data_config = yaml.safe_load(open(data_config_path, 'r'))
        self.sequences = self.data_config["split"][split]
        self.split_ext = SPLIT_FILES[split]
        self.labels = self.data_config['labels']
    
        self.learning_map = self.data_config["learning_map"]
    
        self.learning_map_inv = self.data_config["learning_map_inv"]
        self.color_map = self.data_config['color_map']
    
        self.augmentation = augmentation
        # self.shuffle_index = shuffle_index
        
        self.files = {}
        self.filenames = []
        
        for ext in self.split_ext:
            if self.stage != 1:
                self.files[EXT_TO_NAME[ext]] = []
            self.files["point_input"] = []
            if split != 'test':
                self.files["point_label"] = []
        
        for sequence in self.sequences:
            sequence = str(sequence).zfill(2)
            complete_path = os.path.join(data_root, "sequences", sequence, "voxels")
            if not os.path.exists(complete_path): raise RuntimeError("Voxel directory missing: " + complete_path)
            
            point_path = os.path.join(data_root, "sequences", sequence, "velodyne")
            if not os.path.exists(point_path): raise RuntimeError("Velodyne directory missing: " + point_path)
            
            point_label_path = os.path.join(data_root, "sequences", sequence, "labels")
            if not os.path.exists(point_path): raise RuntimeError("Point labels directory missing: " + point_path)
        
            files = os.listdir(complete_path)
            for ext in self.split_ext:
                data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(ext)])
                if ext == '.bin':
                    point_data = sorted([d.replace('voxels', 'velodyne') for d in data])
                    if len(point_data) == 0: raise RuntimeError("Missing data for " + "point_input")
                    self.files["point_input"].extend(point_data)
                if ext == '.label':
                    point_label = sorted([d.replace('voxels', 'labels') for d in data])
                    if len(point_label) == 0: raise RuntimeError("Missing data for " + "point_label")
                    self.files["point_label"].extend(point_label)
                if self.stage != 1:
                    if len(data) == 0: raise RuntimeError("Missing data for " + EXT_TO_NAME[ext])
                    self.files[EXT_TO_NAME[ext]].extend(data)
            
                    
            # this information is handy for saving the data later, since you need to provide sequences/XX/predictions/000000.label:
            self.filenames.extend(
                sorted([(sequence, os.path.splitext(os.path.basename(f))[0]) for f in files if f.endswith(self.split_ext[0])])
            )
        
        self.num_files = len(self.filenames)
        
        # sanity check:
        for k, v in self.files.items():
            print(k, len(v))
            assert (len(v) == self.num_files)
            
    def augmentation_stage1(self, data):
        # random augmentation
        no_aug = np.random.randint(0, 5)
        if no_aug != 0: return data
    
        flip_type = np.random.randint(0, 4)
        if flip_type==1: # flip x
            data[:, 0] = 51.2 - data[:, 0]
        elif flip_type==2: #flip y
            data[:, 1] = -data[:, 1]
        elif flip_type==3: 
            data[:, 0] = 51.2 - data[:, 0]
            data[:, 1] = -data[:, 1]

        # random data augmentation by rotation
        rotate_rad = torch.deg2rad(torch.rand(1) * 90) - torch.pi / 4
        c, s = torch.cos(rotate_rad), torch.sin(rotate_rad)
        j = torch.tensor([[c, s], [-s, c]])
        data[:, :2] = torch.matmul(data[:, :2], j)

        # random noise scale augmentation 
        noise_scale = np.random.uniform(0.95, 1.05)
        data[:, 0] = noise_scale * data[:, 0]
        data[:, 1] = noise_scale * data[:, 1]
        return data

    def augmentation_stage2(self, data, flip_type, is_scan=False):
        if flip_type==0:
            if is_scan:
                data[:, 1] = -data[:, 1]
            else:
                data = torch.flip(data, dims=(1,))
        return data

    @staticmethod
    def get_remap_lut(learning_map, completion=False):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = max(learning_map.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(learning_map.keys())] = list(learning_map.values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        if completion:
            remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
            remap_lut[0] = 0  # only 'empty' stays 'empty'.
        
        return remap_lut

    @staticmethod
    def get_inv_remap_lut(learning_map_inv):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''
        # make lookup table for mapping
        maxkey = max(learning_map_inv.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(learning_map_inv.keys())] = list(learning_map_inv.values())

        return remap_lut

    
    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)


    def get_n_classes(self):
        return len(self.learning_map_inv)

                   
    def __len__(self):
        return self.num_files

    
    def __getitem__(self, t):
        """ fill dictionary with available data for given index . """
        collection = {}
        if self.stage == 1:
            aug = self.augmentation_stage1
        else:
            aug = partial(
                self.augmentation_stage2, 
                flip_type = np.random.randint(0, 5),
            )
        
        # read raw data and unpack (if necessary)
        for typ in self.files.keys():
            scan_data = None
            if typ == "point_input":
                cyl_voxel_size = (self.cyl_max_extent - self.cyl_min_extent) / (self.cyl_voxel_dims - 1)
                
                # get cordinate and remission infor from raw point
                point = np.fromfile(self.files[typ][t], dtype=np.float32).reshape((-1, 4))
                coord = torch.from_numpy(point[:,:3])
                feat = torch.from_numpy(point[:,3:])
                
                # apply augmentation
                if self.augmentation:
                    if self.stage==1:
                        coord = aug(coord)
                    else:
                        coord = aug(coord, is_scan=True)

                # get label data
                if self.split != 'test':
                    label = np.fromfile(self.files['point_label'][t], dtype=np.uint32)
                    label = label.reshape((-1))  & 0xFFFF # remove instance infor
                    label = SSCDataset.get_remap_lut(self.learning_map, completion=False)[label]
                    label = torch.from_numpy(label.astype('int64'))
                else:
                    label = None

                if self.stage != 1:
                    # filter point in pre-defined range
                    mask = torch.all(
                        ((coord >= self.min_extent) & 
                         (coord < self.max_extent)), 
                        dim=1
                    )
                    coord = coord[mask]
                    feat = feat[mask]
                    if label is not None:
                        label = label[mask]

                # transform to polar coord
                raw_pcoord = cart2polar(coord)
                pcoord = torch.clamp(raw_pcoord, min=self.cyl_min_extent[0], max=self.cyl_max_extent[0])
                
                # voxelize org coord (cube partition)
                quantized_coord = torch.floor((coord-self.min_extent) / 0.2)

                # voxelize polar coord (cylinder partition)
                quantized_pcoord = torch.floor((pcoord - self.cyl_min_extent) / cyl_voxel_size)

                pvcoord, p2v_map, offsets = torch.unique(
                    quantized_pcoord, sorted=True, 
                    return_inverse=True, return_counts=True, dim=0
                )

                
                # pvcoord = quantized_pcoord[[0] + sizes[:-1]]
                center_coord = (quantized_pcoord + 0.5) * cyl_voxel_size + self.cyl_min_extent
                feat = torch.cat([raw_pcoord - center_coord, pcoord, coord, feat], dim=1)
                
                if label is not None:
                    offsets = scatter(torch.ones_like(p2v_map), p2v_map)
                    sizes = offsets.tolist()
                    label_list = torch.split(label, sizes)

                    # pooling point label for voxel label
                    vlabel_list = []
                    for l in label_list:
                        vlabel_list.append(l.mode(dim=0, keepdim=True).values)

                    vlabel = torch.cat(vlabel_list, dim=0)
                    assert vlabel.shape[0] == pvcoord.shape[0]
                
                    collection['point'] = Data(
                        coord = coord,
                        x = feat,
                        y = label,
                        p_coord = pcoord,
                        q_coord = quantized_coord
                        # p2v = p2v_map,
                    )
                    collection['cylinder_label'] = vlabel
                    
                else:
                    collection['point'] = Data(
                        coord = coord,
                        x = feat,
                        p_coord = pcoord,
                        q_coord = quantized_coord
                        # p2v = p2v_map,
                    )
                    
                collection['cylinder_coord'] = pvcoord
                collection['p2v'] = p2v_map
                continue
            elif typ == "label":
                label = np.fromfile(self.files[typ][t], dtype=np.uint16)
                scan_data = SSCDataset.get_remap_lut(self.learning_map, completion=True)[label]
                scan_data = torch.from_numpy(scan_data.astype('int64'))
                scan_data = scan_data.reshape(self.voxel_dims)
                if self.augmentation:
                    scan_data = aug(scan_data, is_scan=False)
                
                # ss_label = SSCDataset.get_remap_lut(self.learning_map, completion=False)[label]
                # ss_label = torch.from_numpy(ss_label.astype('int64'))
                # ss_label = ss_label.reshape(self.voxel_dims)
                # ss_label = self.augmentation_random_flip(ss_label, flip_type, is_scan=False)
                # collection['ss_label'] = ss_label
            elif typ == 'point_label': continue
            else:
                scan_data = torch.from_numpy(unpack(np.fromfile(self.files[typ][t], dtype=np.uint8)).astype('float32'))
                # turn in actual voxel grid representation.
                scan_data = scan_data.reshape(self.voxel_dims)
                if self.augmentation:
                    scan_data = aug(scan_data, is_scan=False)
            
            collection[typ] = scan_data

        quantized_infor = {
            'voxel_dims': self.voxel_dims,
            'min_extent': self.min_extent,
            'max_extent': self.max_extent,
            'cyl_voxel_dims': self.cyl_voxel_dims,
            'cyl_max_extent': self.cyl_max_extent,
            'cyl_min_extent': self.cyl_min_extent,
        }
            
        return self.filenames[t], collection, quantized_infor


    
# if __name__ == "__main__":
#   # Small example of the usage.
#   from pathlib import Path

#   dataset_path = "../semanticKITTI/dataset/"
#   # Replace "/path/to/semantic/kitti/" with actual path to the folder containing the "sequences" folder
#   dataset = SSCDataset(dataset_path)
#   print("# files: {}".format(len(dataset)))

#   (seq, filename), data = dataset[100]
#   print("Contents of entry 100 (" + seq + ":" + filename + "):")
#   for k, v in data.items():
#     print("  {:8} : shape = {}, contents = {}".format(k, v.shape, np.unique(v)))
