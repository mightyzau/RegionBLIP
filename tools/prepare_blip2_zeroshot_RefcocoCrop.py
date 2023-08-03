#coding=utf-8
# 从 RefCOCO Test 中扣出子图，用BLIP2进行captioning测试。

import json 
import os
import cv2
import numpy as np


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--refcoco_test_ann_file', type=str, default='/mnt_jianchong/datasets/lavis/Refer/refcoco/Refcoco_test.json')
    parser.add_argument('-m', '--image_dir', type=str, default='/mnt_jianchong/datasets/lavis/Refer/images')
    parser.add_argument('-o', '--out_dir', type=str, default='/mnt_jianchong/datasets/lavis/RefCOCO_crop_test')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=False)
    os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=False)

    datas = json.load(open(args.refcoco_test_ann_file))

    out_datas_d = {}


    for d in datas:
        caption = d['caption']
        image_path = d['image']
        image_id = d['image_id']

        bbox = np.array(d['bbox'])          # x1y1wh
        bbox[2:] += bbox[:2]                # x1y1x2y2
        x1, y1, x2, y2 = bbox.astype(np.int).tolist()

        image = cv2.imread(os.path.join(args.image_dir, image_path))
        crop = np.ascontiguousarray(image[y1:y2, x1:x2, :])

        out_file = image_id + "_bbox_{}_{}_{}_{}.jpg".format(x1, y1, x2, y2)

        if out_file in out_datas_d:
            out_datas_d[out_file].append(caption)
        else:
            out_datas_d[out_file] = [caption]
            cv2.imwrite(os.path.join(args.out_dir, 'images', out_file), crop)
            print('write: {}'.format(os.path.join(args.out_dir, 'images', out_file)))
    

    out_datas = []
    for img in out_datas_d:
        out_datas.append({'image': img, 'caption': out_datas_d[img], 'image_id': os.path.splitext(os.path.basename(img))[0]})
    
    out_ann_file = os.path.join(args.out_dir, 'RefCOCO_test_crop.json')
    with open(out_ann_file, 'w') as f:
        json.dump(out_datas, f)
    print('write: {}'.format(out_ann_file))
