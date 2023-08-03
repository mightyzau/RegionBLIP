import os 
import logging
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lavis.datasets.datasets.base_dataset import BaseDataset 
from lavis.datasets.pointcloud_utils import (farthest_point_sample, random_point_dropout, random_scale_point_cloud,
                                      shift_point_cloud, rotate_perturbation_point_cloud, rotate_point_cloud)

    
class PointcloudCaptionDataset(Dataset):
    def __init__(self, pc_root, caption_file, 
                 text_processor, 
                 pc_augmentation=True, 
                 num_points=8192,
                 dataset_repeat_num=1):
        super().__init__()
        self.pc_root = pc_root
        self.num_points = num_points
        self.permutation = np.arange(self.num_points)
        self.pc_augmentation = pc_augmentation
        self.text_processor = text_processor

        self.annotations = {}
        with open(caption_file, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                l = l.strip()
                if l == "": continue
                obj_name, *caption = l.split(",")
                caption = ','.join(caption)

                self.annotations[obj_name] = caption
        self.obj_name_list = list(self.annotations.keys())
        assert len(self.annotations) == len(self.obj_name_list)

        self.dataset_repeat_num = dataset_repeat_num
        if self.dataset_repeat_num > 1:
            self.obj_name_list = self.obj_name_list * self.dataset_repeat_num
    
    def __len__(self):
        return len(self.obj_name_list)

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def pc_norm(self, pc):
        # pc: [N, C]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m 
        return pc
    
    def __getitem__(self, index):
        obj_name = self.obj_name_list[index]

        ## read pointcloud
        pc = np.load(os.path.join(self.pc_root, obj_name, "{}_{}.npz".format(obj_name, self.num_points)))
        pc = pc["arr_0"]    # [8192, 3]

        if self.num_points < pc.shape[0]:
            pc = farthest_point_sample(pc, self.num_points)
        else:
            pc = self.random_sample(pc, self.num_points)
        
        pc = self.pc_norm(pc)
        if self.pc_augmentation:
            pc = random_point_dropout(pc[None, ...])
            pc = random_scale_point_cloud(pc)
            pc = shift_point_cloud(pc)
            pc = rotate_perturbation_point_cloud(pc)
            pc = rotate_point_cloud(pc)
            pc = pc.squeeze(0)

        caption = self.text_processor(self.annotations[obj_name])

        return {
            "text_input": caption,
            "pointcloud": pc,
            "obj_name": obj_name,
            'image_id': obj_name
        }
    
    def collater(self, samples):
        return default_collate(samples)
    