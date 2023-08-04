#coding=utf-8

import os 
import json
import logging
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_file', type=str)
    parser.add_argument('-a', '--annotation_file', type=str)
    args = parser.parse_args()

    logging.info("--" * 10)
    with open(args.annotation_file, 'r') as f:
        gts = json.load(f)
        logging.info("ground-truth: {} samples".format(len(gts['images'])))
    with open(args.result_file, 'r') as f:
        preds = json.load(f)
        logging.info("predictions: {} samples".format(len(preds)))


    coco = COCO(args.annotation_file)
    coco_result = coco.loadRes(args.result_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
