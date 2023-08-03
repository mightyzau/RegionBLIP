#coding=utf-8
# Convert Cap3D_automated_Objaverse_test to coco-caption format for evaluation.

import os
import json


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_gt_file', type=str, default="/mnt_jianchong/datasets/lavis/Cap3D/Cap3D_automated_Objaverse_test.txt")
    args = parser.parse_args()

    annotations = []
    images = []

    id = 0
    with open(args.input_gt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == "": continue
            obj_name, *caption = line.split(',')
            caption = ','.join(caption)

            images.append({'id': obj_name})
            annotations.append({'image_id': obj_name, 'id': id + 1, 'caption': caption})
            id += 1
    
    outs = {'annotations': annotations, 'images': images}
    outfile = os.path.splitext(args.input_gt_file)[0] + '__jianchong.json'

    with open(outfile, 'w') as f:
        json.dump(outs, f, indent=4)
    print('write to {}'.format(outfile))
    