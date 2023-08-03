#coding=utf-8
"""
CUDA_VISIBLE_DEVICES=0 python run_scripts/regionblip/tools/demo_ImageRegion.py \
    --cfg-path run_scripts/regionblip/eval_opt/opt_6.7b/eval_regionblip_opt_ImageRegion-caption_refcoco.yaml \
    --options model.load_pretrained=True model.pretrained=training_dir/regionblip_opt_task_ImageRegionText_PointcloudText_PointcloudRegionText__opt-6.7b__A10040G__epoch-30__gpu-8/checkpoint_29.pth
"""

import os, sys
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '../', '../'))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import argparse
import pickle
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode, get_world_size
from lavis.datasets.datasets.dataloader_utils import PrefetchLoader
from lavis.common.logger import setup_logger
from lavis.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str)

    parser.add_argument("--pdb_debug", action='store_true')

    parser.add_argument("--options", nargs="+",
                        help="override some settings in the used config, the key-value pair "
                            "in xxx=yyy format will be merged into config file (deprecate), "
                            "change to --cfg-options instead.")
    args = parser.parse_args()
    return args 

def setup_seeds(config):
    seed = config.run_cfg.seed

    random.seed(seed )
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benckmark = False 
    cudnn.deterministic = True 

def build_model(cfg):
    model_config = cfg.model_cfg 
    model_cls = registry.get_model_class(model_config.arch)
    return model_cls.from_config(model_config)

def build_datasets(cfg):
    datasets_config = cfg.datasets_cfg
    dataset_names = list(datasets_config.keys())
    assert len(dataset_names) == 1, "One dataset to be evaluated."
    
    dataset_name = dataset_names[0]
    dataset_config = datasets_config[dataset_name]

    builder = registry.get_builder_class(dataset_name)(dataset_config)
    dataset = builder.build_datasets()

    split_names = cfg.run_cfg.test_splits
    assert len(split_names) == 1
    split_name = split_names[0]
    dataset = dataset[split_name]

    logging.info("Dataset {} Split {}, loaded {} records.".format(dataset_name, split_name, len(dataset)))

    return dataset

def build_dataloader(cfg, dataset):
    if cfg.run_cfg.distributed:
        sampler = DistributedSampler(
            dataset, shuffle=False, num_replicas=get_world_size(), rank=get_rank())
        if not cfg.run_cfg.get('use_dist_eval_sampler', True):
            sampler = None
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.run_cfg.batch_size_eval,
        num_workers=cfg.run_cfg.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        collate_fn=getattr(dataset, 'collater', None),
        drop_last=False
    )
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def pc_norm(pc):
    # pc: [N, C]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m 
    return pc


def main():
    args = parse_args()
    if args.pdb_debug:
        import pdb; pdb.set_trace()
    
    cfg = Config(args)
    if args.pdb_debug:
        cfg.run_cfg['num_workers'] = 0
    
    cfg.run_cfg.batch_size_eval = 1
    
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    cfg.pretty_print()

    dataset = build_datasets(cfg)
    dataloader = build_dataloader(cfg, dataset)

    model = build_model(cfg)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for samples in dataloader:
            caption_pred = model.generate_pointcloudregion_caption_posembed(
                samples,
                use_nucleus_sampling=False,
                num_beams=cfg.run_cfg.num_beams,
                max_length=cfg.run_cfg.max_len,
                min_length=cfg.run_cfg.min_len,
                num_captions=1
            )

            datas = {}
            datas['point_clouds'] = samples['point_clouds'].cpu().numpy()
            datas['refer_box_corners'] = samples['refer_box_corners'].cpu().numpy()
            datas['point_colors'] = samples['point_colors'].cpu().numpy()
            datas['text'] = samples['text']

            with open('pc_region_data.pkl', 'wb') as f:
                pickle.dump(datas, f)

            print('write to pc_region_data.pkl')

            print('gt_caption: {}'.format(samples['text']))
            print('pred_caption: {}'.format(caption_pred))

            print('*' * 10)



if __name__ == '__main__':
    main()
    