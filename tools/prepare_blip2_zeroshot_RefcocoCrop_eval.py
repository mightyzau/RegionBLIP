
import json 
import random
import os


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--ann_file', type=str, default="/mnt_jianchong/datasets/lavis/RefCOCO_crop_test/RefCOCO_test_crop.json")
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    datas = json.load(open(args.ann_file))

    random.seed(args.seed)
    random.shuffle(datas)

    images = []
    annotations = []

    instance_id = 0
    for d in datas:
        image_id = d['image_id']
        images.append({'id': image_id})

        for cap in d['caption']:
            instance_id += 1
            annotations.append({
                    'image_id': image_id,
                    'id': instance_id,
                    'caption': cap
                })
    
    out_file = os.path.splitext(args.ann_file)[0] + '__jianchong.json'
    with open(out_file, 'w') as f:
        json.dump({'images': images, 'annotations': annotations}, f)
    print('write: {}'.format(out_file))

    print('  {} images'.format(len(images)))
    print('  {} annotations'.format(len(annotations)))
