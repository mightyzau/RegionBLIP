import logging
import torch.distributed as dist

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, load_dataset_config
from lavis.datasets.datasets.scannet_reference_datasets import ScannetReferenceDatasest
from lavis.common.dist_utils import is_main_process, is_dist_avail_and_initialized


@registry.register_builder("scannet_refer")
class ScannetReferBuilder(BaseDatasetBuilder):
    train_dataset_cls = ScannetReferenceDatasest
    eval_dataset_cls = ScannetReferenceDatasest

    DATASET_CONFIG_DICT = {
        'default': 'configs/datasets/scannet/defaults_cap.yaml'}

    def __init__(self, cfg):
        if isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            self.config = cfg 
    
    def build_datasets(self):
        if is_dist_avail_and_initialized():
            dist.barrier()
        logging.info("Building datasets ...")
        datasets = self.build()
        return datasets
    
    def build(self):
        datasets = dict()
        annotations = self.config.annotations
        for split in annotations.keys():
            assert split in ['train', 'val', 'test']
            is_train = split == 'train'
            annotation_info = annotations.get(split)

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            augment = True if is_train else False 

            dataset_repeat_num = annotation_info.get('dataset_repeat_num', 1)

            datasets[split] = dataset_cls(
                scene_root_dir=annotation_info.scene_root_dir,
                scanner_file=annotation_info.scanner_file,
                scannet_v2_tsv=annotation_info.scannet_v2_tsv,
                num_points=annotation_info.num_points,
                use_color=annotation_info.use_color,
                use_height=annotation_info.use_height,
                augment=augment,
                use_random_cuboid=annotation_info.use_random_cuboid,
                dataset_repeat_num=dataset_repeat_num)
        return datasets
    