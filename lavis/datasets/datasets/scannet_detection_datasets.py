# Refer to https://github.com/facebookresearch/3detr

import os
import numpy as np 
import torch 
import logging

from torch.utils.data import Dataset
import lavis.models.detr3d.utils.pc_util as pc_util
from lavis.models.detr3d.utils.pc_util import scale_points, shift_scale_points
from lavis.models.detr3d.utils.random_cuboid import RandomCuboid
from lavis.models.detr3d.utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor, 
                                                get_3d_box_batch_np, get_3d_box_batch_tensor)
from torch.utils.data.dataloader import default_collate


IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = ""  ## Replace with path to dataset
DATASET_METADATA_DIR = "" ## Replace with path to dataset


class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1  # 默认是 axis-aligned 3d-boxes
        self.max_num_obj = 64

        self.type2class = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "showercurtrain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "garbagebin": 17,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        # Semantic Segmentation Classes. Not used in 3DETR
        self.num_class_semseg = 20
        self.type2class_semseg = {
            "wall": 0,
            "floor": 1,
            "cabinet": 2,
            "bed": 3,
            "chair": 4,
            "sofa": 5,
            "table": 6,
            "door": 7,
            "window": 8,
            "bookshelf": 9,
            "picture": 10,
            "counter": 11,
            "desk": 12,
            "curtain": 13,
            "refrigerator": 14,
            "showercurtrain": 15,
            "toilet": 16,
            "sink": 17,
            "bathtub": 18,
            "garbagebin": 19,
        }
        self.class2type_semseg = {
            self.type2class_semseg[t]: t for t in self.type2class_semseg
        }
        self.nyu40ids_semseg = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class_semseg = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        }

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)      # Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)
    

class ScannetDetectionDataset(Dataset):
    def __init__(self, split='train', 
                       root_dir=None, 
                       meta_data_dir=None,
                       num_points=40000,
                       use_color=False,
                       use_height=False,
                       augment=False,
                       use_random_cuboid=True,
                       random_cuboid_min_points=30000):
        self.config = ScannetDatasetConfig()
        assert split in ['train', 'val']

        if split == 'val':
            augment = False

        self.data_path = root_dir
        all_scan_names = list(
            set([os.path.basename(x)[0:12] for x in os.listdir(self.data_path) if x.startswith("scene")])
        )
        
        split_filename = os.path.join(meta_data_dir, "scannetv2_{}.txt".format(split))
        with open(split_filename, "r") as f:
            self.scan_names = f.read().splitlines()
        # remove unavailable scans
        num_scans = len(self.scan_names)
        self.scan_names = [name for name in self.scan_names if name in all_scan_names]
        logging.info("keep {} scans out of {}".format(len(self.scan_names), num_scans))

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.use_random_cubioid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, "{}_vert.npy".format(scan_name)))
        instane_labels = np.load(os.path.join(self.data_path, "{}_ins_label.npy".format(scan_name)))
        semantic_labels = np.load(os.path.join(self.data_path, "{}_sem_label.npy".format(scan_name)))
        instance_bboxes = np.load(os.path.join(self.data_path, "{}_bbox.npy".format(scan_name)))

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height 
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)


        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.config.max_num_obj 
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        if self.augment and self.use_random_cubioid:
            point_cloud, instance_bboxes, per_point_labels = self.random_cuboid_augmentor(point_cloud, instance_bboxes, [instane_labels, semantic_labels])
            instane_labels = per_point_labels[0]
            semantic_labels = per_point_labels[1]
        
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        instane_labels = instane_labels[choices]
        semantic_labels = semantic_labels[choices]

        sem_seg_labels = np.ones_like(semantic_labels) * IGNORE_LABEL

        for _c in self.config.nyu40ids_semseg:
            semantic_labels[semantic_labels == _c] = self.config.nyu40id2class_semseg[_c]
        
        pcl_color = pcl_color[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1 
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.config.rotate_aligned_boxes(
                target_bboxes, rot_mat)

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
                                    box_centers[None, ...],
                                    src_range=[point_cloud_dims_min[None, ...],
                                               point_cloud_dims_max[None, ...]],
                                    dst_range=self.center_normalizing_range)
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min 
        box_sizes_normalized = scale_points(
                        raw_sizes.astype(np.float32)[None, ...],
                        mult_factor=1.0 / mult_factor[None, ...])
        box_sizes_normalized = box_sizes_normalized.squeeze(0)
        
        box_corners = self.config.box_parametrization_to_corners_np(
                box_centers[None, ...],
                raw_sizes.astype(np.float32)[None, ...],
                raw_angles.astype(np.float32)[None, ...])
        box_corners = box_corners.squeeze(0)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0 : instance_bboxes.shape[0]] = [
            self.config.nyu40id2class[int(x)]
            for x in instance_bboxes[:, -1][0:instance_bboxes.shape[0]]]

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)

        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["pcl_color"] = pcl_color
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)

        ret_dict["scan_name"] = scan_name
        return ret_dict

    def collater(self, samples):
        batch = len(samples)

        scan_name_list = []
        for b in range(batch):
            scan_name = samples[b].pop('scan_name')
            scan_name_list.append(scan_name)

        outputs = default_collate(samples)
        outputs['num_semcls'] = [self.config.num_semcls] * batch
        outputs['num_angle_bin'] = [self.config.num_angle_bin] * batch

        classnames = [self.config.class2type[i] for i in range(self.config.num_semcls)]
        outputs['classnames'] = ["|".join(classnames)] * batch
        outputs['scan_name'] = scan_name_list
        return outputs
    