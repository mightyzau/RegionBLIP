#coding=utf-8
# Convert refcoco_test to coco-caption format for evaluation.


import json 
import os 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--original_gt_file', type=str, default="/mnt_jianchong/datasets/lavis/ScanRefer_Dataset/raw/ScanRefer_filtered_val.json")
    args = parser.parse_args()

    #import pdb; pdb.set_trace()

    datas = json.load(open(args.original_gt_file))

    annotations = []
    images = []
    all_image_ids = []
    for i, d in enumerate(datas):
        scene_id = d['scene_id']
        object_id = d['object_id']
        scene_id = "{}_{}".format(scene_id, object_id)

        if scene_id not in all_image_ids:
            all_image_ids.append(scene_id)
            images.append({'id': scene_id})

        annotations.append({'image_id': scene_id, 'id': i + 1, 'caption': d['description']})
    
    outs = {'annotations': annotations, 'images': images}

    outfile = os.path.splitext(args.original_gt_file)[0] + '__jianchong.json'

    with open(outfile, 'w') as f:
        json.dump(outs, f, indent=4)
    print('write to {}'.format(outfile))

    assert len(all_image_ids) == len(images)
    print('  {} images'.format(len(images)))
    print('  {} instances'.format(len(annotations)))
