import os
import logging
import numpy as np
import json
from torch.utils.data import Dataset 
from torch.utils.data.dataloader import default_collate

from lavis.models.detr3d.utils.random_cuboid import RandomCuboid
import lavis.models.detr3d.utils.pc_util as pc_util
from lavis.models.detr3d.utils.pc_util import scale_points, shift_scale_points
from lavis.datasets.datasets.scannet_detection_datasets import ScannetDatasetConfig


MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


class ScannetReferenceDatasest(Dataset):
    def __init__(self, 
                 scene_root_dir="",
                 scanner_file="ScanRefer_filtered_train.json",
                 scannet_v2_tsv="scannetv2-labels.combined.tsv",
                 num_points=40000,
                 use_height=False,
                 use_color=False, 
                 augment=False,
                 use_random_cuboid=True,
                 random_cuboid_min_points=30000,
                 dataset_repeat_num=1):
        self.config = ScannetDatasetConfig()
        self.scene_root_dir = scene_root_dir
        self.scannet_v2_tsv = scannet_v2_tsv
        self.scanrefer = json.load(open(scanner_file)) 
        # list of scanrefer anns. each ann has the keys: dict_keys(['scene_id', 'object_id', 'object_name', 'ann_id', 'description', 'token'])
        assert min([int(d['object_id']) for d in self.scanrefer]) == 0, "object_id should start from 0"

        self.dataset_repeat_num = dataset_repeat_num
        if self.dataset_repeat_num > 1:
            logging.info('dataset_repeat_num: {}'.format(self.dataset_repeat_num))
            self.scanrefer = self.scanrefer * self.dataset_repeat_num

        self.num_points = num_points
        self.use_height = use_height
        self.use_color = use_color
        #self.use_normal = use_normal
        self.augment = augment
        self.use_random_cubioid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

    def __len__(self):
        return len(self.scanrefer)
    
    def _get_data(self, index):
        scene_id = self.scanrefer[index]['scene_id']                                # "scene0041_00"
        object_id = int(self.scanrefer[index]['object_id'])                         # 34
        #object_name = " ".join(self.scanrefer[index]["object_name"].split("_"))     # 'chair'
        #ann_id = self.scanrefer[index]["ann_id"]
        caption = self.scanrefer[index]['description']

        # get point cloud
        mesh_vertices = np.load(os.path.join(self.scene_root_dir, "{}_aligned_vert.npy".format(scene_id)))      # [50000, 9]
        instance_labels = np.load(os.path.join(self.scene_root_dir, "{}_ins_label.npy".format(scene_id)))       # [50000,]
        semantic_labels = np.load(os.path.join(self.scene_root_dir, "{}_sem_label.npy".format(scene_id)))       # [50000,]
        instance_bboxes = np.load(os.path.join(self.scene_root_dir, "{}_aligned_bbox.npy".format(scene_id)))    # [#bbox, 8]
        object_ids = [int(d)  for d in instance_bboxes[:, -1].tolist()]
        #assert min(object_ids) == 0, "object_ids ({}) should starts from 0".format(object_ids)
        return scene_id, object_id, caption, mesh_vertices, instance_labels, semantic_labels, instance_bboxes, object_ids
    
    def __getitem__(self, index):
        """
        ScanRefer data format:
            "scene_id": [ScanNet scene id, e.g. "scene0000_00"],
            "object_id": [ScanNet object id (corresponds to "objectId" in ScanNet aggregation file), e.g. "34"], 0-indexed.
            "object_name": [ScanNet object name (corresponds to "label" in ScanNet aggregation file), e.g. "coffee_table"],
            "ann_id": [description id, e.g. "1"],
            "description": [...],
            "token": [a list of tokens from the tokenized description] 
        """
        scene_id, object_id, caption, mesh_vertices, instance_labels, semantic_labels, instance_bboxes, object_ids = self._get_data(index)
        if object_id not in object_ids:
            logging.info("scene_id {}: object_id {} is not in object_ids: {}".format(scene_id, object_id, object_ids))
            # reset index
            index = 0
            scene_id, object_id, mesh_vertices, instance_labels, semantic_labels, instance_bboxes, object_ids = self._get_data(index)
        ref_box_index = object_ids.index(object_id)

        #if not self.use_color:
        #    point_cloud = mesh_vertices[:, 0:3]
        #else:
        #    point_cloud = mesh_vertices[:, 0:6]
        #    point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
        point_cloud = mesh_vertices[:, 0:6]
        point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
        
        #if self.use_normal:
        #    normals = mesh_vertices[:, 6:9]
        #    point_cloud = np.concatenate([point_cloud, normals], axis=1)
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            heigit = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, heigit[:, None]], 1)

        if self.augment and self.use_random_cubioid:
            point_cloud, instance_bboxes, per_point_labels = self.random_cuboid_augmentor(point_cloud, instance_bboxes, [instance_labels, semantic_labels])
            instance_labels = per_point_labels[0]
            semantic_labels = per_point_labels[1]
        
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                instance_bboxes[:, 0] = -1 * instance_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                instance_bboxes[:, 1] = -1 * instance_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            instance_bboxes = self.config.rotate_aligned_boxes(instance_bboxes, rot_mat)

        raw_sizes = instance_bboxes[:, 3:6]                                                 # [#boxes, 3]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]
        box_centers = instance_bboxes.astype(np.float32)[:, 0:3]                            # [#boxes, 3]

        #box_centers_normalized = shift_scale_points(
        #                            box_centers[None, ...],
        #                            src_range=[point_cloud_dims_min[None, ...],
        #                                       point_cloud_dims_max[None, ...]],
        #                            dst_range=self.center_normalizing_range)
        #box_centers_normalized = box_centers_normalized.squeeze(0)
        #mult_factor = point_cloud_dims_max - point_cloud_dims_min 
        #box_sizes_normalized = scale_points(
        #                raw_sizes.astype(np.float32)[None, ...],
        #                mult_factor=1.0 / mult_factor[None, ...])
        #box_sizes_normalized = box_sizes_normalized.squeeze(0)
        
        #raw_angles = np.zeros(len(instance_bboxes), dtype=np.float32)
        #box_corners = self.config.box_parametrization_to_corners_np(
        #        box_centers[None, ...],
        #        raw_sizes.astype(np.float32)[None, ...],
        #        raw_angles.astype(np.float32)[None, ...])
        #box_corners = box_corners.squeeze(0)

        if True:
            l = np.expand_dims(raw_sizes[..., 0], -1)  # [#boxes, 1]
            w = np.expand_dims(raw_sizes[..., 1], -1)
            h = np.expand_dims(raw_sizes[..., 2], -1)

            corners_3d = np.zeros([len(box_centers), 8, 3])
            corners_3d[..., :, 0] = np.concatenate(
            (l / 2, l / 2, -l / 2, -l / 2,    l / 2, l / 2, -l / 2, -l / 2), -1
            )
            corners_3d[..., :, 1] = np.concatenate(
            (h / 2, h / 2, h / 2, h / 2,      -h / 2, -h / 2, -h / 2, -h / 2), -1
            )
            corners_3d[..., :, 2] = np.concatenate(
            (w / 2, -w / 2, -w / 2, w / 2,    w / 2, -w / 2, -w / 2, w / 2), -1
            )
            corners_3d += np.expand_dims(box_centers, -2)

            box_corners = corners_3d
        
        point_colors = point_cloud[:, 3:6] * 256.0 + MEAN_COLOR_RGB
        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        
        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)                                                   # [40000, 3]
        ret_dict["point_colors"] = point_colors.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)                                  # [3,]
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)                                  # [3,]

        ret_dict["refer_box_corners"] = box_corners.astype(np.float32)[ref_box_index]                               # [8, 3]
        ret_dict["refer_box_id"] = "{}_{}".format(scene_id, object_id)
        ret_dict['text'] = caption

        return ret_dict
    
    def collater(self, samples):
        batch = len(samples)
        outputs = default_collate(samples)
        outputs['num_semcls'] = [self.config.num_semcls] * batch
        outputs['num_angle_bin'] = [self.config.num_angle_bin] * batch

        return outputs
    