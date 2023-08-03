"""
For Objaverse test set.

CUDA_VISIBLE_DEVICES=0 python tools/demo_ImageRegion.py \
    --cfg-path run_scripts/regionblip/eval_opt/opt_6.7b/eval_regionblip_opt_Pointcloud-caption_objaverse.yaml \
    --options model.load_pretrained=True model.enable_pointcloud_qformer=True \
             model.pretrained=training_dir/regionblip_opt_task_ImageRegionText_PointcloudText_PointcloudRegionText__opt-6.7b__A10040G__epoch-30__gpu-8/checkpoint_29.pth
"""

import os, sys
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import argparse
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lavis.common.config import Config
from lavis.common.logger import setup_logger
from lavis.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str)

    parser.add_argument('--pc_dir', type=str, default="/mnt_jianchong/datasets/lavis/objaverse_triplet/objaverse_pc_parallel/")
    parser.add_argument('--pc_ann_file', type=str, default="/mnt_jianchong/datasets/lavis/Cap3D/Cap3D_automated_Objaverse_test.txt")

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
    
    setup_seeds(cfg)
    setup_logger()

    cfg.pretty_print()

    model = build_model(cfg)
    model.cuda()
    model.eval()

    annotations = {}
    with open(args.pc_ann_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l = l.strip()
            if l == "": continue
            obj_name, *caption = l.split(",")
            caption = ','.join(caption)

            annotations[obj_name] = caption
    obj_name_list = list(annotations.keys())

    for index in range(50):
        obj_name = obj_name_list[index]

        pc = np.load(os.path.join(args.pc_dir, obj_name, "{}_{}.npz".format(obj_name, 8192)))
        pc = pc["arr_0"]    # [8192, 3]
        pc = pc_norm(pc)
        caption = annotations[obj_name]

        caption_pred = model.generate_pointcloud_caption(
            {"pointcloud": torch.as_tensor(pc, dtype=torch.float32).cuda().unsqueeze(0)},
            use_nucleus_sampling=False,
            num_beams=cfg.run_cfg.num_beams,
            max_length=cfg.run_cfg.max_len,
            min_length=cfg.run_cfg.min_len,
            num_captions=1
        )

        print('*' * 20)
        print('obj_name: {}'.format(obj_name))
        print('  gt_caption: {}'.format(caption))
        print('  pred_caption: {}'.format(caption_pred))


if __name__ == '__main__':
    main()
    