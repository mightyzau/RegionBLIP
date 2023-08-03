"""
For RefCOCO test set.

CUDA_VISIBLE_DEVICES=1 python run_scripts/regionblip/tools/demo_Image.py 、
    --cfg-path run_scripts/regionblip/eval_opt/opt_6.7b/eval_regionblip_opt_Image-caption_coco.yaml \
    --options model.load_pretrained=True model.enable_pointcloud_qformer=True \
        model.pretrained=training_dir/regionblip_opt_task_ImageRegionText_PointcloudText_PointcloudRegionText__opt-6.7b__A10040G__epoch-30__gpu-8/checkpoint_29.pth
"""

import os, sys
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '../', '../'))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import cv2
import json
from PIL import Image
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from lavis.common.config import Config
from lavis.common.logger import setup_logger
from lavis.common.registry import registry
from lavis.models.deformable_detr.util import box_ops


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str)

    parser.add_argument('--test_img_dir', type=str, default="/mnt_jianchong/datasets/lavis/coco/images/")
    parser.add_argument('--test_ann_file', type=str, default="/mnt_jianchong/datasets/lavis/coco/annotations/coco_karpathy_test.json")

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

def build_processor(cfg):
    coco_caption_cfg = cfg.datasets_cfg.coco_caption
    vis_proc_cfg = coco_caption_cfg.get('vis_processor').get('eval')
    txt_proc_cfg = coco_caption_cfg.get('text_processor').get('eval')

    vis_processor = registry.get_processor_class(vis_proc_cfg.name).from_config(vis_proc_cfg)
    txt_processor = registry.get_processor_class(txt_proc_cfg.name).from_config(txt_proc_cfg)

    return vis_processor, txt_processor


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

    vis_processor, txt_processor = build_processor(cfg)


    img_dir = args.test_img_dir
    annotations = json.load(open(args.test_ann_file))


    for index in range(50):
        ann = annotations[index]
        image_path = os.path.join(img_dir, ann['image'])
        image = Image.open(image_path).convert('RGB')

        image = vis_processor(image)       # [3, H, W]
        caption = ann['caption']


        caption_pred = model.generate_image_caption(
            {"image": torch.as_tensor(image, dtype=torch.float32).cuda().unsqueeze(0)},
            use_nucleus_sampling=False,
            num_beams=cfg.run_cfg.num_beams,
            max_length=cfg.run_cfg.max_len,
            min_length=cfg.run_cfg.min_len,
            num_captions=1
        )

        print('*' * 20)
        print('{}'.format(image_path))
        print('  gt_caption: {}'.format(caption))
        print('  pred_caption: {}'.format(caption_pred))


if __name__ == '__main__':
    main()
    