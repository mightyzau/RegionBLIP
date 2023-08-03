"""
For RefCOCO test.

CUDA_VISIBLE_DEVICES=0 python tools/demo_ImageRegion.py \
    --cfg-path run_scripts/regionblip/eval_opt/opt_6.7b/eval_regionblip_opt_ImageRegion-caption_refcoco.yaml \
    --options model.load_pretrained=True \
    model.pretrained=training_dir/regionblip_opt_task_ImageRegionText_PointcloudText_PointcloudRegionText__opt-6.7b__A10040G__epoch-30__gpu-8/checkpoint_29.pth
"""


import os, sys
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str)

    parser.add_argument('--test_img_dir', type=str, default="/mnt_jianchong/datasets/lavis/Refer/images/")
    parser.add_argument('--test_ann_file', type=str, default="/mnt_jianchong/datasets/lavis/Refer/refcoco/Refcoco_test.json")

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
    general_box_caption_cfg = cfg.datasets_cfg.general_box_caption
    vis_proc_cfg = general_box_caption_cfg.get('vis_processor').get('eval')
    txt_proc_cfg = general_box_caption_cfg.get('text_processor').get('eval')

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


    for index, ann in enumerate(annotations):

        # note, these images has more than one boxes to captioning.
        if ann['image_id'] not in ['coco_93581', 'coco_383885', 'coco_136836', 'coco_460986', 'coco_461561', 'coco_78322', 'coco_141947', 'coco_204792', 'coco_208820', 'coco_48442', 'coco_427606', 'coco_75800', 'coco_184684', 'coco_170785', 'coco_65420', 'coco_563123', 'coco_62743', 'coco_194223', 'coco_449808', 'coco_547797', 'coco_199836', 'coco_472312', 'coco_488373']:
            continue

        image_path = os.path.join(img_dir, ann['image'])
        image = Image.open(image_path).convert('RGB')
        vis_image = cv2.imread(image_path)

        w, h = image.size 
        box = torch.as_tensor(ann['bbox'], dtype=torch.float32).reshape(4) # x1y1wh
        box[2:] += box[:2]    # x1y1wh to x1y1x2y2
        box[0::2].clamp_(min=0, max=w)
        box[1::2].clamp_(min=0, max=h)
        vis_image = cv2.rectangle(vis_image, box[:2].type(torch.int).tolist(), box[2:].type(torch.int).tolist(), (0, 0, 255), 2)

        # normalize to [0, 1]
        box = box.type(torch.float)
        box[0::2] /= w 
        box[1::2] /= h

        image = vis_processor(image)       # [3, H, W]
        caption = txt_processor(ann['caption'])


        caption_pred = model.generate_imageregion_caption_posembed(
            {"image": torch.as_tensor(image, dtype=torch.float32).cuda().unsqueeze(0),
            'box': box.view(1, 4).cuda()},
            use_nucleus_sampling=False,
            num_beams=cfg.run_cfg.num_beams,
            max_length=cfg.run_cfg.max_len,
            min_length=cfg.run_cfg.min_len,
            num_captions=1
        )

        print('*' * 20)
        print('index: {}'.format(index))
        print('  gt_caption: {}'.format(caption))
        print('  pred_caption: {}'.format(caption_pred))

        vis_file = "vis_image_region_images/{}_index-{}.png".format(ann['image_id'], index)
        os.makedirs(os.path.dirname(vis_file), exist_ok=True)
        cv2.imwrite(vis_file, vis_image)


if __name__ == '__main__':
    main()
    