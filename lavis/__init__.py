"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

from omegaconf import OmegaConf

from lavis.common.registry import registry

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.tasks import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])


# add dataset paths for regionblip
registry.register_path('caption_evaluate_refcoco_test',
                       '/mnt_jianchong2/datasets/Refer/refcoco/Refcoco_test__jianchong.json')

registry.register_path('caption_evaluate_objaverse_test',
                       '/mnt_jianchong2/datasets/Cap3D/Cap3D_automated_Objaverse_test__jianchong.json')

registry.register_path('caption_evaluate_scanrefer_val',
                       '/mnt_jianchong2/datasets/ScanRefer_Dataset/raw/ScanRefer_filtered_val__jianchong.json')
