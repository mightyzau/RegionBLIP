
# RegionBLIP: A Unified Multi-modal Pre-training Framework for Holistic and Regional Comprehension

Qiang Zhou, Chaohui Yu, Shaofeng Zhang, Sitong Wu, Zhibing Wang, Fan Wang

**Alibaba Damo Academy**

<a href='https://regionblip.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'>  <a href='https://arxiv.org/abs/xxx'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


## Introduction
---
<p align="center">
    <img src=./assets/regionblip.png  width="800">
<p>

RegionBLIP is a unified multi-modal pre-training framework for holistic and regional comprehension. RegionBILP is pretrained incrementally, freezing the Q-Former from BLILP-2, and optimizing a set of modality-specific Lora parameters in Q-Former and LLM.

<br/>

Examples of captioning results for images, image-regions, point-clouds, and point-cloud-regions are shown below.

<p align="center">
    <img src=./assets/examples.png  width="800">
<p>


<br/>
We provide the pre-trained RegionBLIP models below.

| Model | I-region captioning (RefCOCO test) | PCD captioning (Objaverse test) | P-region captioning (ReferScannet val) |
| --- | --- | --- | --- |
| [RegionBLIP-OPT-2.7B](https://drive.google.com/file/d/1YS6XuRh3plH6i8VP5qgU0g2grOxsyvkY/view?usp=drive_link) | 63.5 | 112.7 | 57.0 |
| [RegionBLIP-OPT-6.7B](https://drive.google.com/file/d/1_Q3AVVFocBOPHUXiLJAwHvP0mG8XU98H/view?usp=drive_link) | 64.2 | 113.6 | 59.3 |
| [RegionBLIP-T5-XL](https://drive.google.com/file/d/1raJmYJbZh2KRoPY6hgJy8meyFrzQ0WkP/view?usp=drive_link) | 47.6 | 108.1 | 59.2 |
| RegionBLILP-T5-XXL | 56.1 | 109.0 | 53.6 |



## Installation
---
We recommend using conda to create the environment.

```
conda create -n regionblip python=3.8 -y
conda activate regionblip

conda install pytorch=1.10 torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install loralib termcolor plyfile trimesh -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Additional packages installed for the point encoder.
```
pip install scikit-learn ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
git clone https://github.com/unlimblue/KNN_CUDA
cd KNN_CUDA
python setup.py build install [--user]

# NOTE:* 如果遇到 "from knn_cuda import KNN" 的错误，大概率是安装问题。可以直接指定安装路径 in the line-6 of lavis.model.pointbert.dvae.py。

```


## Datasets
---

##### Image-Text Data
We test directly on the COCO-caption dataset. To download the COCO-caption dataset, please refer to [this script](lavis/datasets/download_scripts/download_coco.py).


<br/>

##### Image-Region-Text Data
For the image-region-text data, we use the RefCOCO dataset. Please refer to this [repository](https://github.com/lichengunc/refer) to download annotation files. The image files are just the MSCOCO images.

The downloaded data looks like:
```
|-- Refer
|    |-- images
|    |    |-- train2014
|    |    |-- val2014
|    |    |-- test2014
|    |    |-- test2015
|    |-- refcoco
|    |    |-- Refcoco_train.json
|    |    |-- Refcoco_test.json
|    |    |-- Refcoco_test__jianchong.json
```

<br/>

##### Pointcloud-Text Data
For the point cloud data, we use the released Objaverse dataset from [ULIP](https://github.com/salesforce/ULIP). For the corresponding captions, we use the released text data from [Cap3D](https://huggingface.co/datasets/tiange/Cap3D).

The downloaded data looks like:
```
|-- objaverse_text
|    |-- ULIP-Objaverse_triplets
|    |    |-- objaverse_pc_parallel
|    |-- Cap3D
|    |    |-- Cap3D_automated_Objaverse_train.txt
|    |    |-- Cap3D_automated_Objaverse_test.txt
|    |    |-- Cap3D_automated_Objaverse_test__jianchong.json
```

<br/>

##### Pointcloud-Region-Text Data
For the pointcloud-region-text data, we refer to the [ScanRefer](https://github.com/daveredrum/ScanRefer) for dataset prepartion. 

The ScanRefer data can be downloaded from the [opendatalab](https://opendatalab.com/ScanRefer_Dataset/cli) and the downloaded data looks like:

```
|-- scanref_data
|    |-- ScanRefer_Dataset
|    |    |-- raw
|    |    |    |-- ScanRefer_filtered_train.json
|    |    |    |-- ScanRefer_filtered_val.json
|    |    |    |-- ScanRefer_filtered_val__jianchong.json
|    |-- scannet_reference_data
|    |    |-- meta_data
|    |    |-- scannet_data
```

To prepre the point cloud data, first download the ScanNet_v2 dataset from [opendatalab](https://opendatalab.com/ScanNet_v2/download). Then preprocess the ScanNet data according to [ScanRefer](https://github.com/daveredrum/ScanRefer) to generate the required *scannet_data* folder.

```
ln -s /mnt_jianchong/datasets/lavis/ScanNet_v2/raw/scans ./
python batch_load_scannet_data.py
```

## Training
---
The training scripts are all located at *run_scripts/train/*.

```
bash run_scripts/train/regionblip_opt-2.7b_freeze-qformer_regloss_pcqformer_A10040G_4node.sh
```
For exampe, the above script is used to train RegionBLIP-OPT-2.7B with four nodes of 8×A100 machines. The same model can also be pre-trained with only one node machine, and the pre-training time is slightly longer.


## Test
---
We provide scripts to convert test annotation files to the format required by *COCOEvalCap*.
```
# Generate Refcoco_test__jianchong.json
python tools/prepare_regionblip_eval_refercoco.py -i Refer/refcoco/Refcoco_test.json


# Generate ScanRefer_filtered_val__jianchong.json
python tools/prepare_regionblip_eval_referscannet.py -i ScanRefer_Dataset/raw/ScanRefer_filtered_val.json


# Generate Cap3D_automated_Objaverse_test__jianchong.json
python tools/prepare_regionblip_eval_objaverse.py -i Cap3D/Cap3D_automated_Objaverse_test.txt
```
The paths to these transformed ground-truth files should be registered in [*lavis/_\_init\__.py*](lavis/__init__.py).


<br/>

For example, to evaluate *RegionBLIP-OPT-2.7B*'s image-region captioning performance, run the following command with downloaded weight of *regionblip_opt-2.7b.pth*.

```
python -m torch.distributed.run --nproc_per_node=8 evaluate.py \
    --cfg-path run_scripts/eval_opt/eval_regionblip_opt-2.7b_ImageRegion-caption_refcoco__A10040G.yaml \
    --options model.load_pretrained=True model.pretrained=regionblip_opt-2.7b.pth
```

The result should look like this.
```
SPICE: 0.214
Bleu_1: 0.337
Bleu_2: 0.191
Bleu_3: 0.106
Bleu_4: 0.057
METEOR: 0.245
ROUGE_L: 0.391
CIDEr: 0.635
SPICE: 0.214
```


## RegionCap-10M Dataset

We also release our collected [RegionCap-10M](https://drive.google.com/drive/folders/19iTkmWkRsXgxcbuJ9Ce7PqCyQjAnvY1P?usp=drive_link) dataset.


## Acknowledgement
---
If you're using RegionBLIP in your research or applications, please cite using this BibTeX:
```
@article{zhou2023regionblip,
  title={RegionBLIP: A Unified Multi-modal Pre-training Framework for Holistic and Regional Comprehension},
  author={Qiang Zhou, Chaohui Yu, Shaofeng Zhang, Sitong Wu, Zhibing Wang, Fan Wang},
  journal={arXiv preprint arXiv:TO-REPLACE},
  year={2023}
}
```


## License
---
This repository is under BSD 3-Clause License. Many codes are based on [LAVIS](https://github.com/salesforce/LAVIS) with BSD 3-Clause License.