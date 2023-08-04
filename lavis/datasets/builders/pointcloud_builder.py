import os
import warnings
import logging
import torch.distributed as dist

import lavis.common.utils as utils
from lavis.common.registry import registry
from lavis.common.dist_utils import is_main_process, is_dist_avail_and_initialized
from lavis.datasets.builders.base_dataset_builder import (BaseDatasetBuilder, 
                                load_dataset_config, BaseProcessor)
from lavis.datasets.datasets.pointcloud_datasets import MobileNet40Dataset


@registry.register_builder('mobilenet40')
class MobileNet40Builder(BaseDatasetBuilder):
    train_dataset_cls = MobileNet40Dataset
    eval_dataset_cls = MobileNet40Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pointcloud/defaults_mobilenet40.yaml"
    }

    def __init__(self, cfg):
        if isinstance(cfg, str):
            self.config = load_dataset_config(cfg) 
        else:
            self.config = cfg 

        self.pointcloud_processor = {'train': BaseProcessor(), 'eval': BaseProcessor()}
    
    
    def build_processors(self):
        pass 

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        pc_info = build_info.pointcloud 

        datasets = dict()
        for split in ['train', 'test']:

            is_train = split == 'train'

            pc_processor = (
                self.pointcloud_processor['train']
                if is_train
                else self.pointcloud_processor['eval'])
            
            pc_path = pc_info.storage
            if not os.path.isabs(pc_path):
                pc_path = utils.get_cache_path(pc_path)
            if not os.path.exists(pc_path):
                warnings.warn('storage path {} does not exist.'.format(pc_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                point_processor=pc_processor,
                point_root=pc_path,
                npoints=pc_info.npoints,
                split=split
            )
        
        return datasets
    
    def _download_data(self):
        pass 

    def build_datasets(self):
        if is_main_process():
            self._download_data()
        
        if is_dist_avail_and_initialized():
            dist.barrier()
        
        logging.info("Building datasets ...")
        datasets = self.build()

        return datasets        
