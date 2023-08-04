import os
import logging
import pickle
import numpy as np


from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.pointcloud_utils import farthest_point_sample

class MobileNet40Dataset(BaseDataset):
    def __init__(self, point_processor, point_root, npoints=8192, split='test'):
        self.point_processor = point_processor
        self.root = point_root
        self.npoints = npoints
        self.permutation = np.arange(self.npoints)
        assert split in ['train', 'test']

        self.point_path = os.path.join(self.root, 'modelnet40_{}_{}pts_fps.dat'.format(split, self.npoints))
        assert os.path.isfile(self.point_path), 'please preprocess (sampling points) as ULIP.'
        with open(self.point_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)
        logging.info("MobileNet40Dataset, load processed pointcloud from {}".format(self.point_path))
        self.list_of_instances = [i for i in range(len(self.list_of_labels))]

        shape_name_file = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.classnames = [line.strip() for line in open(shape_name_file) if line.strip() != ""]

        self.templates = templates_modelnet40   # used for zero-shot classification

    def __len__(self):
        return len(self.list_of_labels)

    def pc_norm(self, pc):
        # pc: [N, C]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m 
        return pc
    
    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def __getitem__(self, index):
        points, label = self.list_of_points[index], self.list_of_labels[index]
        #label_name = self.classnames[int(label.item())]
        instance_id = self.list_of_instances[index]

        points = points[:, :3]
        if self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)
        else:
            points = self.random_sample(points, self.npoints)
        
        points = self.pc_norm(points)
        pointcloud = self.point_processor(points)

        return {
            "pointcloud": pointcloud,
            "label": int(label.item()),
            "instance_id": instance_id
        }


templates_modelnet40 = [
    "a point cloud model of {}.",
    "There is a {} in the scene.",
    "There is the {} in the scene.",
    "a photo of a {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of a {}.",
    "itap of my {}.",
    "itap of the {}.",
    "a photo of a {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of a {}.",
    "a good photo of the {}.",
    "a bad photo of a {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of a {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of a {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of a {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of a {}",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}."
]