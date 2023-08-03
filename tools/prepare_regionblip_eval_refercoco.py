#coding=utf-8
# Convert refcoco_test to coco-captioning format for evaluation.


import json 
import os 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--original_gt_file', type=str, default="/mnt_jianchong/datasets/lavis/Refer/refcoco/Refcoco_test.json")
    args = parser.parse_args()

    datas = json.load(open(args.original_gt_file))

    annotations = []
    images = []
    all_image_ids = []
    instance_id = 0

    for d in datas:
        image_id = d['image_id']
        bbox = d['bbox']
        image_id = "{}_box_{}_{}_{}_{}".format(image_id, *[int(x) for x in bbox])

        if image_id in all_image_ids:
            pass 
        else:
            images.append({'id': image_id})
            all_image_ids.append(image_id)

        annotations.append({'image_id': image_id, 'id': instance_id, 'caption': d['caption']})
        instance_id += 1
    
    outs = {'annotations': annotations, 'images': images}

    outfile = os.path.splitext(args.original_gt_file)[0] + '__jianchong.json'

    with open(outfile, 'w') as f:
        json.dump(outs, f, indent=4)
    print('write to {}'.format(outfile))
    print('  {} images'.format(len(images)))
    print('  {} instances'.format(len(annotations)))
