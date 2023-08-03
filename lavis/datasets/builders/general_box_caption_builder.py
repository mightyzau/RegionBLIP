import os
import logging
import torch.distributed as dist

import lavis.common.utils as utils
from lavis.common.registry import registry
from lavis.common.dist_utils import is_dist_avail_and_initialized
from lavis.datasets.datasets.general_box_caption_datasets import GeneralBoxCapProcessor, GeneralBoxCaptionDataset, GeneralBoxCaptionEvalDataset
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, load_dataset_config


@registry.register_builder("general_box_caption")
class GeneralBoxCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = GeneralBoxCaptionDataset
    eval_dataset_cls = GeneralBoxCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/boxcap/defaults.yaml"
    }

    def __init__(self, cfg):
        if isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            self.config = cfg 
        
        self.vis_processors = {}
        self.text_processors = {}

    def build_datasets(self):
        if is_dist_avail_and_initialized():
            dist.barrier()
        
        logging.info("Building datasets ...")
        datasets = self.build()

        return datasets 
    
    def build(self):
        self.build_processors()

        build_info = self.config.build_info 
        ann_info = build_info.annotations
        img_info = build_info.images

        datasets = dict()
        for split in ann_info.keys():
            if split not in ['train', 'val', 'test']:
                logging.info('Not supported split: {}\n'.format(split))
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            ann_path = ann_info.get(split).storage 
            if not os.path.isabs(ann_path):
                ann_path = utils.get_cache_path(ann_path)
            
            image_root = img_info.storage
            if not os.path.isabs(image_root):
                image_root = utils.get_cache_path(image_root)
            
            dataset_repeat_num = ann_info.get(split).get('dataset_repeat_num', 1)
            
            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                caption_file=ann_path,
                image_root=image_root,
                dataset_repeat_num=dataset_repeat_num,
            )
        
        return datasets


@registry.register_builder("general_box_caption_2")
class GeneralBoxCapBuilder2(GeneralBoxCapBuilder):
    train_dataset_cls = GeneralBoxCaptionDataset
    eval_dataset_cls = GeneralBoxCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/boxcap/defaults_2.yaml"
    }