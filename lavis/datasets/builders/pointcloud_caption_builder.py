import os 
import logging
import torch.distributed as dist

import lavis.common.utils as utils
from lavis.common.registry import registry
from lavis.common.dist_utils import is_main_process, is_dist_avail_and_initialized
from lavis.datasets.builders.base_dataset_builder import load_dataset_config, BaseDatasetBuilder, BaseProcessor
from lavis.datasets.datasets.pointcloud_caption_datasets import PointcloudCaptionDataset



@registry.register_builder("pointcloud_caption")
class PointcloudCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = PointcloudCaptionDataset
    eval_dataset_cls = PointcloudCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/defaults_cap.yaml"
    }

    def __init__(self, cfg):
        if isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            self.config = cfg 
        
        self.text_processors = {"train": BaseProcessor, "eval": BaseProcessor}

        txt_proc_cfg = self.config.get("text_processor")
        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get('train')
            txt_eval_cfg = txt_proc_cfg.get('eval')

            if txt_train_cfg is not None:
                self.text_processors['train'] = self._build_proc_from_cfg(txt_train_cfg)
            if txt_eval_cfg is not None:
                self.text_processors['eval'] = self._build_proc_from_cfg(txt_eval_cfg)

    def build_datasets(self):
        if is_dist_avail_and_initialized():
            dist.barrier()
        logging.info("Building datasets ...")
        datasets = self.build()

        return datasets

    def build(self):
        build_info = self.config.build_info

        ann_info = build_info.annotations
        pc_info = build_info.pointcloud

        datasets = dict()
        for split in ann_info.keys():
            assert split in ['train', 'val', 'test']

            is_train = split == 'train'
            text_processor = self.text_processors["train"] if is_train else self.text_processors['eval']
            ann_path = ann_info.get(split).storage
            pc_path = pc_info.storage

            dataset_repeat_num = ann_info.get('dataset_repeat_num', 1)

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                pc_root=pc_path,
                caption_file=ann_path,
                text_processor=text_processor,
                pc_augmentation=pc_info.pc_augmentation,
                num_points=pc_info.num_points,
                dataset_repeat_num=dataset_repeat_num)
        return datasets
