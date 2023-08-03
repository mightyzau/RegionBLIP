import os
import json
import logging
from PIL import Image
import torch 
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


@registry.register_processor("general_box_caption_img_trainval")
class GeneralBoxCapProcessor(BaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean, std),]
        )
    
    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


class GeneralBoxCaptionDataset(Dataset):
    def __init__(self, vis_processor, text_processor, image_root, caption_file, dataset_repeat_num=1):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.image_root = image_root
    
        self.annotations = json.load(open(caption_file))

        self.dataset_repeat_num = dataset_repeat_num
        if self.dataset_repeat_num > 1:
            logging.info('dataset_repeat_num: {}'.format(self.dataset_repeat_num))
            self.annotations = self.annotations * self.dataset_repeat_num

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        w, h = image.size 

        box = torch.torch.as_tensor(ann['bbox'], dtype=torch.float32).reshape(4) # x1y1wh
        box[2:] += box[:2]    # x1y1wh to x1y1x2y2
        box[0::2].clamp_(min=0, max=w)
        box[1::2].clamp_(min=0, max=h)
        box = box.type(torch.int)

        box_attn_mask = torch.zeros((h, w))
        box_attn_mask[box[1]:box[3], box[0]:box[2]] = 1.0

        # normalize to [0, 1]
        box = box.type(torch.float)
        box[0::2] /= w 
        box[1::2] /= h

        image = self.vis_processor(image)       # [3, H, W]
        nh, nw = image.size()[-2:]
        box_attn_mask = transforms.functional.resize(box_attn_mask.unsqueeze(0), (nh, nw), interpolation=InterpolationMode.BICUBIC).squeeze(0)

        caption = self.text_processor(ann['caption'])

        if 'image_id' not in ann:
            ann['image_id'] = os.path.basename(ann['image'])

        return {
            "image": image,
            "box": box,
            "box_attn_mask": box_attn_mask,
            "caption": caption,
            "image_id": ann['image_id']
        }
    
    def collater(self, samples):
        image_list, box_list, caption_list, box_attn_mask_list, image_id_list = [], [], [], [], []
        for sample in samples:
            image_list.append(sample['image'])
            box_list.append(sample['box'])
            caption_list.append(sample['caption'])
            box_attn_mask_list.append(sample['box_attn_mask'])
            image_id_list.append(sample['image_id'])
        
        images = torch.stack(image_list, dim=0)
        box_attn_masks = torch.stack(box_attn_mask_list, dim=0)
        boxes = torch.stack(box_list, dim=0)

        return {
            'image': images,
            'box_attn_mask': box_attn_masks,
            'box': boxes,
            'text_input': caption_list,
            'image_id': image_id_list}


class GeneralBoxCaptionEvalDataset(GeneralBoxCaptionDataset):
    def __init__(self, vis_processor, text_processor, image_root, caption_file, dataset_repeat_num=1):
        assert dataset_repeat_num == 1
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.image_root = image_root

        annotations = json.load(open(caption_file))

        # box 去重
        self.annotations = []
        self.image_ids = {}

        for ann in annotations:
            img_id = ann['image_id']
            box = ann['bbox']
            img_id = "{}_box_{}_{}_{}_{}".format(img_id, *[int(x) for x in box])
            if img_id in self.image_ids:
                continue
            else:
                self.image_ids[img_id] = 1
                ann['image_id'] = img_id
                self.annotations.append(ann)
