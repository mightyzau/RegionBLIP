# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
from dataclasses import dataclass
from packaging import version
from typing import Optional
import contextlib

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
import transformers
from transformers import T5TokenizerFast
from transformers.modeling_outputs import ModelOutput

from lavis.common.utils import is_url
from lavis.common.dist_utils import download_cached_file
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from .Qformer import BertConfig, BertLMHeadModel
from .modeling_t5 import T5Config, T5ForConditionalGeneration


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

@dataclass
class AlignLLM_Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None

@dataclass
class AlignLLMRegion_Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None
    loss_reg: Optional[torch.FloatTensor] = None


@registry.register_model("regionblip_t5")
class RegionBLIPT5(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enable_pointcloud_qformer = cfg.get('enable_pointcloud_qformer', False)
        
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer(bert_path=cfg.get('bert_path', None))  


        self.visual_encoder, self.ln_vision = self.init_vision_encoder(cfg.vit)
        if cfg.get('freeze_vit', False):
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        
        self.point_encoder = self.init_pointcloud_encoder(cfg.pointbert)
        if cfg.get('freeze_point_encoder', False):
            for name, param in self.point_encoder.named_parameters():
                param.requires_grad = False
            self.point_encoder = self.point_encoder.eval()
            self.point_encoder.train = disabled_train
            logging.info("freeze point encoder !!!")
        self.pc_adapter = nn.Linear(self.point_encoder.trans_dim, cfg.qformer.input_dim)

        
        self.Qformer, self.Qformer_pc, self.query_tokens_img, self.query_tokens_img_region, self.query_tokens_pc, self.query_tokens_pc_region = self.init_Qformer(
            cfg.qformer.num_query_token, self.visual_encoder.num_features, bert_path=cfg.get('bert_path', None), enable_pointcloud_qformer=self.enable_pointcloud_qformer)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        if self.enable_pointcloud_qformer:
            self.Qformer_pc.resize_token_embeddings(len(self.tokenizer))

        if cfg.get('freeze_qformer', False):
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            num_lora_params = 0
            for name ,param in self.Qformer.named_parameters():
                param.requires_grad = False 
                if 'lora_' in name:
                    param.requires_grad = True
                    num_lora_params += param.numel()
            logging.info("freeze Qformer, leaving {} lora trainable params".format(num_lora_params))
    
                
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(cfg.t5_model)              
        t5_config = T5Config.from_pretrained(cfg.t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            cfg.t5_model, config=t5_config
        )
        num_lora_params = 0
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            if self.cfg.get('t5_precision') == "bfloat16":
                param.data = param.data.bfloat16()
            elif self.cfg.get('t5_precision') == "float16":
                param.data = param.data.float16()
            elif self.cfg.get('t5_precision') == 'float':
                param.data = param.data.float()
            else:
                raise ValueError('not supported t5_precision: {}'.format(self.cfg.get('t5_precision')))
            
            if 'lora_' in name:
                param.requires_grad = True
                param.data = param.data.float()
                num_lora_params += param.numel()
        logging.info("T5 model has {} lora trainable params".format(num_lora_params))
                

        ## For stage-2, adapter query features to LLM
        self.t5_proj_img = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
        self.t5_proj_img_region = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
        self.t5_proj_pc = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
        self.t5_proj_pc_region = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)


        ## For stage-1 of modal alignment, adapter query features to qformer text embedding
        self.stage1_proj_text = nn.Linear(self.Qformer.config.hidden_size, cfg.embed_dim)
        self.stage1_proj_img = nn.Linear(self.Qformer.config.hidden_size, cfg.embed_dim)
        self.stage1_proj_img_region = nn.Linear(self.Qformer.config.hidden_size, cfg.embed_dim)
        self.stage1_proj_pc = nn.Linear(self.Qformer.config.hidden_size, cfg.embed_dim)
        self.stage1_proj_pc_region = nn.Linear(self.Qformer.config.hidden_size, cfg.embed_dim)
        self.img_itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))


        self.max_txt_len = cfg.get("max_txt_len", 32)                                                       # 32
        self.prompt = cfg.get('prompt', "")                                                                 # "a photo of"
  
        self._apply_lemmatizer = cfg.get('apply_lemmatizer', False)                                         # False
        self._lemmatizer = None  


        ## For posembed assisted cross-attenion
        self.img_region_posembed = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, self.query_tokens_img_region.size(-1)))
        
        self.pc_region_posembed = nn.Sequential(
            nn.Linear(24, 128),
            nn.GELU(),
            nn.Linear(128, self.query_tokens_pc_region.size(-1)))   


        if cfg.get('freeze_blip2_pretrain', False):
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info('freeze self.ln_vision')

            self.query_tokens_img.requires_grad = False
            self.query_tokens_img.train = disabled_train
            logging.info('freeze self.query_tokens_img')

            for name, param in self.t5_proj_img.named_parameters():
                param.requires_grad = False
            self.t5_proj_img = self.t5_proj_img.eval()
            self.t5_proj_img.train = disabled_train
            logging.info('freeze self.t5_proj_img')

            num_lora_params = 0
            for name ,param in self.Qformer.named_parameters():
                if name.endswith('lora_A_img') or name.endswith('lora_B_img'):
                    param.requires_grad = False
                    num_lora_params += param.numel()
            logging.info('Freeze Qformer lora_A_img and lora_B_img: {}'.format(num_lora_params))

            num_lora_params = 0
            for name ,param in self.t5_model.named_parameters():
                if name.endswith('lora_A_img') or name.endswith('lora_B_img'):
                    param.requires_grad = False
                    num_lora_params += param.numel()
            logging.info('Freeze t5_model lora_A_img and lora_B_img: {}'.format(num_lora_params))
        

        if self.cfg.get('enable_aux_regloss', False):
            self.reg_img_region = nn.Linear(self.Qformer.config.hidden_size, 4)
            self.reg_pc_region = nn.Linear(self.Qformer.config.hidden_size, 24)


    @classmethod
    def init_tokenizer(cls, truncation_side="right", bert_path=None):
        from transformers import BertTokenizer
        if bert_path is not None:
            tokenizer = BertTokenizer.from_pretrained(bert_path, truncation_side=truncation_side)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def init_vision_encoder(self, cfg):
        model_name = cfg.get('vit_model', "eva_clip_g")
        img_size = cfg.img_size
        drop_path_rate = cfg.get('drop_path_rate', 0) 
        use_grad_checkpoint = cfg.get('use_grad_checkpoint', False)
        precision = cfg.get('vit_precision', 'fp16')

        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def init_pointcloud_encoder(self, cfg):
        from lavis.models.pointbert.point_encoder import PointTransformer
        point_encoder = PointTransformer(pretrain_weight=cfg.pretrained, with_color=cfg.get('with_color', False))
        return point_encoder
    
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, bert_path=None, enable_pointcloud_qformer=False):
        if bert_path is not None:
            encoder_config = BertConfig.from_pretrained(bert_path)
        else:
            encoder_config = BertConfig.from_pretrained("bert-base-uncased")

        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        if bert_path is not None:
            Qformer = BertLMHeadModel.from_pretrained(
                bert_path, config=encoder_config
            )
        else:
            Qformer = BertLMHeadModel.from_pretrained(
                "bert-base-uncased", config=encoder_config
            )
        if enable_pointcloud_qformer:
            if bert_path is not None:
                Qformer_pc = BertLMHeadModel.from_pretrained(
                    bert_path, config=encoder_config
                )
            else:
                Qformer_pc = BertLMHeadModel.from_pretrained(
                    "bert-base-uncased", config=encoder_config
                )
        else:
            Qformer_pc = None

        query_tokens_img = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))                                # [1, 32, 768]
        query_tokens_img.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        query_tokens_img_region = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))                                # [1, 32, 768]
        query_tokens_img_region.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        query_tokens_pc = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))                                # [1, 32, 768]
        query_tokens_pc.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        query_tokens_pc_region = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))                                # [1, 32, 768]
        query_tokens_pc_region.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, Qformer_pc, query_tokens_img, query_tokens_img_region, query_tokens_pc, query_tokens_pc_region
    
    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """

        if cfg.get('load_blip2_pretrain', False):
            pretrain_path = cfg.get('blip2_pretrain', None)
            assert os.path.isfile(pretrain_path)
            self.load_from_pretrained(url_or_filename=pretrain_path)


        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            load_pretrained = cfg.get("load_pretrained", True)
            if load_pretrained:
                # load pre-trained weights
                pretrain_path = cfg.get("pretrained", None)
                assert "Found load_finetuned is False, but pretrain_path is None."
                self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


    def forward(self, samples):
        task_name = samples['task_name']

        if task_name == 'task_alignLLM_ImageText':
            return self.forward_image(samples)
        elif task_name == 'task_alignLLM_ImageRegionText_posembed':
            return self.forward_image_region(samples)
        
        elif task_name == 'task_alignLLM_PointcloudText':
            return self.forward_pointcloud(samples)
        elif task_name == 'task_alignLLM_PointcloudRegionText_posembed':
            return self.forward_pointcloud_region(samples)
        
        else:
            raise ValueError('Not supported task name: {}'.format(task_name))

    def forward_image(self, samples):
        image = samples["image"]                                                                        # [batch, 3, 224, 224]
        text_input = samples["text_input"]                                                             

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))                                   # [batch, 257, 1408]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device)                                                                              

        query_tokens = self.query_tokens_img.expand(image_embeds.shape[0], -1, -1)                    
        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
                modal_name='image')                                                                   
            # query_output.last_hidden_state: [batch, 32, 768]

        loss_itc, loss_itm, loss_lm = self._forward_modal_align(vision_embeds=image_embeds,
                                                            query_tokens=query_tokens,
                                                            query_output=query_output,
                                                            text_input=text_input,
                                                            modal_name='image',
                                                            modal_proj=self.stage1_proj_img)
        loss_llm = self._forward_t5_generate(query_output, text_input, self.t5_proj_img, modal_name="image")

        return AlignLLM_Output(
            loss=loss_itc + loss_itm + loss_lm + loss_llm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_llm=loss_llm
        )
    
    def forward_image_region(self, samples):
        image = samples['image']
        box = samples['box']
        assert box.max() <= 1.0
        text_input = samples['text_input']

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        box_query_tokens = self.query_tokens_img_region.expand(image_embeds.size(0), -1, -1)
        box_pos_embeds = self.img_region_posembed(box)
        box_query_tokens = box_query_tokens + box_pos_embeds.unsqueeze(1)

        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                query_embeds=box_query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
                modal_name="image_region"
            )

        loss_itc, loss_itm, loss_lm = self._forward_modal_align(vision_embeds=image_embeds,
                                                            query_tokens=box_query_tokens,
                                                            query_output=query_output,
                                                            text_input=text_input,
                                                            modal_name='image_region',
                                                            modal_proj=self.stage1_proj_img_region)
        loss_llm = self._forward_t5_generate(query_output, text_input, self.t5_proj_img_region, modal_name="image_region")

        if self.cfg.get('enable_aux_regloss', False):
            box_pred = self.reg_img_region(query_output.last_hidden_state.mean(dim=1))      # [batch, 4]
            loss_reg = F.l1_loss(box_pred, box) * self.cfg.get('aux_regloss_weight', 1.0)
        else:
            loss_reg = torch.tensor(0., requires_grad=False).to(self.device)


        return AlignLLMRegion_Output(
            loss=loss_itc + loss_itm + loss_lm + loss_llm + loss_reg,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_llm=loss_llm,
            loss_reg=loss_reg
        )
    
    def forward_pointcloud(self, samples):
        text_input = samples['text_input']
        pc = samples['pointcloud']

        with self.maybe_autocast():
            pc_embeds = self.point_encoder(pc)
            pc_embeds = self.pc_adapter(pc_embeds)
            pc_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc.device)
        
        query_tokens = self.query_tokens_pc.expand(pc_embeds.size(0), -1, -1)

        if self.enable_pointcloud_qformer:
            qformer = self.Qformer_pc
        else:
            qformer = self.Qformer
        with self.maybe_autocast():
            query_output = qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=pc_atts,
                use_cache=True,
                return_dict=True,
                modal_name="pointcloud"
            )
        
        loss_itc, loss_itm, loss_lm = self._forward_modal_align(vision_embeds=pc_embeds,
                                                            query_tokens=query_tokens,
                                                            query_output=query_output,
                                                            text_input=text_input,
                                                            modal_name='pointcloud',
                                                            modal_proj=self.stage1_proj_pc)
        loss_llm = self._forward_t5_generate(query_output, text_input, self.t5_proj_pc, modal_name="pointcloud")

        return AlignLLM_Output(
            loss=loss_itc + loss_itm + loss_lm + loss_llm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_llm=loss_llm
        )
    
    def forward_pointcloud_region(self, samples):
        pc = samples['point_clouds']
        text_input = samples['text']

        with self.maybe_autocast():
            pc_embeds = self.point_encoder(pc)
            pc_embeds = self.pc_adapter(pc_embeds)
            pc_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc.device)
        
        refer_box_corners = samples['refer_box_corners']                                                # [batch, 8, 3]
        #assert refer_box_corners.max() <= 1.0
        pc_pos_embedding = self.pc_region_posembed(refer_box_corners.view(-1, 24))

        query_tokens = self.query_tokens_pc_region.expand(pc_embeds.size(0), -1, -1)
        query_tokens = query_tokens + pc_pos_embedding.unsqueeze(1)

        if self.enable_pointcloud_qformer:
            qformer = self.Qformer_pc
        else:
            qformer = self.Qformer
        with self.maybe_autocast():
            query_output = qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=pc_atts,
                use_cache=True,
                return_dict=True,
                modal_name="pointcloud_region") 
        
        loss_itc, loss_itm, loss_lm = self._forward_modal_align(vision_embeds=pc_embeds,
                                                            query_tokens=query_tokens,
                                                            query_output=query_output,
                                                            text_input=text_input,
                                                            modal_name='pointcloud_region',
                                                            modal_proj=self.stage1_proj_pc_region)
        
        loss_llm = self._forward_t5_generate(query_output, text_input, self.t5_proj_pc_region, modal_name="pointcloud_region")

        if self.cfg.get('enable_aux_regloss', False):
            box_pred = self.reg_pc_region(query_output.last_hidden_state.mean(dim=1))      # [batch, 24]
            loss_reg = F.l1_loss(box_pred, refer_box_corners.flatten(1, 2)) * self.cfg.get('aux_regloss_weight', 1.0)
        else:
            loss_reg = torch.tensor(0., requires_grad=False).to(self.device)

        return AlignLLMRegion_Output(
            loss=loss_itc + loss_itm + loss_lm + loss_llm + loss_reg,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_llm=loss_llm,
            loss_reg=loss_reg
        )

    def _forward_modal_align(self, vision_embeds, query_tokens, query_output, text_input, modal_name, modal_proj):
        
        assert modal_name in ['image', 'image_region', 'pointcloud', 'pointcloud_region']
        if (modal_name in ['pointcloud', 'pointcloud_region']) and self.enable_pointcloud_qformer:
            qformer = self.Qformer_pc
        else:
            qformer = self.Qformer

        # query_output.last_hidden_state: [batch, 32, 768]
        vision_feats = F.normalize(modal_proj(query_output.last_hidden_state), dim=-1)          
        bs = vision_feats.size(0)

        # compute text embeddings
        text_tokens = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)                                                                       
        text_output = qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            modal_name=modal_name)                                                                  
        # text_output.last_hidden_state: [batch, 32, 768]
        text_feat = F.normalize(
            self.stage1_proj_text(text_output.last_hidden_state[:, 0, :]), dim=-1)                    
        

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(vision_feats)                                       # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)                                            # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            vision_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()                                                                             # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)                                                            # [batch_size, batch_size*num_gpu]
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp                                                              

        #rank = dist.get_rank()
        rank = get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)      

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2


        ###============== Image-text Matching ===================###
        if modal_name == 'image':
            text_input_ids_world = concat_all_gather(text_tokens.input_ids)                      
            text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)               
            image_embeds_world = all_gather_with_grad(vision_embeds)                               
            with torch.no_grad():
                weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
                weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
                weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

            # select a negative image for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = query_tokens[0:1, ...].expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(self.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            image_embeds_all = torch.cat(
                [vision_embeds, image_embeds_neg, vision_embeds], dim=0
            )  # pos, neg, pos
            image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(self.device)

            output_itm = qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
                modal_name=modal_name
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
            vl_output = self.img_itm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(self.device)
            loss_itm = F.cross_entropy(logits, itm_labels)
        else:
            loss_itm = torch.tensor(0., requires_grad=False).to(self.device)


        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()                                                  
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100)                                        

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device)                                                                                  
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)                     
        lm_output = qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
            modal_name=modal_name)                                                                      

        loss_lm = lm_output.loss

        return loss_itc, loss_itm, loss_lm
    
    def _forward_t5_generate(self, query_output, text_input, t5_proj, modal_name=""):
        # For encoder-decoder-based LLMs, we pre-train with the prefix language modeling loss, 
        # where we split a text into two parts.

        if modal_name in ['image', 'image_region']:
            prefix = 'a photo of'
        elif modal_name in ['pointcloud', 'pointcloud_region']:
            prefix = 'a point cloud of'
        else:
            raise ValueError('not supported modal_name: {}'.format(modal_name))

        #prefix_text = []
        #suffix_text = []
        #for text in text_input:
        #    words = text.split(' ')
        #    n = len(words) // 2
        #    prefix_text.append(' '.join(words[:n]))
        #    suffix_text.append(' '.join(words[n:]))

        prefix_text = []
        suffix_text = []
        for text in text_input:
            prefix_text.append(prefix)
            suffix_text.append(text)

        inputs_t5 = t5_proj(query_output.last_hidden_state)                                          
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(self.device)             

        t5_precision = self.cfg.get('t5_precision')
        if t5_precision == "bfloat16":
            dtype = torch.bfloat16
        elif t5_precision == "float16":
            dtype = torch.float16
        elif t5_precision == 'float':
            dtype = torch.float32
        else:
            raise ValueError('not supported t5_precision: {}'.format(t5_precision))

        with self.maybe_autocast(dtype=dtype):
            input_tokens = self.t5_tokenizer(
                prefix_text,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            output_tokens = self.t5_tokenizer(
                suffix_text,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
                modal_name=modal_name
            )
            loss = outputs.loss


        return loss


    @torch.no_grad()
    def generate_image_caption(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        image = samples['image']
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        return self._generate_caption(
            samples=samples, embeds=image_embeds, atts=image_atts, 
            t5_proj=self.t5_proj_img, query_tokens=self.query_tokens_img, modal_name="image",
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_captions=num_captions,
            temperature=temperature)
    
    @torch.no_grad()
    def generate_imageregion_caption_posembed(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        image = samples['image']
        box = samples['box']
        assert box.max() <= 1.0

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_pos_embedding = self.img_region_posembed(box).unsqueeze(1)

        return self._generate_caption(
            samples=samples, embeds=image_embeds, atts=image_atts, 
            t5_proj=self.t5_proj_img_region, query_tokens=self.query_tokens_img_region, modal_name="image_region",
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_captions=num_captions,
            temperature=temperature,
            query_pos_embedding=query_pos_embedding)

    @torch.no_grad()
    def generate_pointcloud_caption(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        pc = samples['pointcloud'].float()
        with self.maybe_autocast():
            pc_embed = self.point_encoder(pc)
            pc_embed = self.pc_adapter(pc_embed)
        pc_atts = torch.ones(pc_embed.size()[:-1], dtype=torch.long).to(pc.device)

        return self._generate_caption(
            samples=samples, embeds=pc_embed, atts=pc_atts, 
            t5_proj=self.t5_proj_pc, query_tokens=self.query_tokens_pc, modal_name="pointcloud",
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_captions=num_captions,
            temperature=temperature)
    
    @torch.no_grad()
    def generate_pointcloudregion_caption_posembed(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        pc = samples['point_clouds']
        with self.maybe_autocast():
            pc_embeds = self.point_encoder(pc)
            pc_embeds = self.pc_adapter(pc_embeds)
        pc_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc.device)

        refer_box_corners = samples['refer_box_corners']
        #if self.cfg.get('ablation_no_posembed', False):
        #    refer_img_region_posembed = None
        #else:
        pc_pos_embedding = self.pc_region_posembed(refer_box_corners.view(-1, 24)).unsqueeze(1)

        return self._generate_caption(
            samples=samples, embeds=pc_embeds, atts=pc_atts, 
            t5_proj=self.t5_proj_pc_region, query_tokens=self.query_tokens_pc_region, modal_name="pointcloud_region",
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_captions=num_captions,
            temperature=temperature,
            query_pos_embedding=pc_pos_embedding)
        
    def _generate_caption(
        self,
        samples,
        embeds,
        atts,
        t5_proj,
        query_tokens,
        modal_name,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        query_pos_embedding=None
    ): 
        assert modal_name in ['image', 'image_region', 'pointcloud', 'pointcloud_region']
        if (modal_name in ['pointcloud', 'pointcloud_region']) and self.enable_pointcloud_qformer:
            qformer = self.Qformer_pc
        else:
            qformer = self.Qformer

        with self.maybe_autocast():
            query_tokens = query_tokens.expand(embeds.size(0), -1, -1)     
            if query_pos_embedding is not None:  
                query_tokens = query_tokens + query_pos_embedding
            query_output = qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=embeds,
                encoder_attention_mask=atts,
                return_dict=True,
                modal_name=modal_name)
            
        inputs_t5 = t5_proj(query_output.last_hidden_state)                              # [batch, #tokens, 2560]
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(
            embeds.device) 
            
        #if "prompt" in samples.keys():
        #    prompt = samples["prompt"]
        #else:
        #    prompt = self.prompt      
        if modal_name in ['image', 'image_region']:
            prompt = 'a photo of'
        elif modal_name in ['pointcloud', 'pointcloud_region']:
            prompt = 'a point cloud of'
        else:
            raise ValueError('not supported modal_name: {}'.format(modal_name))
        
        if isinstance(prompt, str):
            prompt = [prompt] * embeds.size(0)
        else:
            assert len(prompt) == embeds.size(
                0
            ), "The number of prompts must be equal to the batch size."
        
        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(self.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1) 

        self.t5_model.modal_name = modal_name
        self.t5_model.encoder.modal_name = modal_name
        self.t5_model.decoder.modal_name = modal_name

        t5_precision = self.cfg.get('t5_precision')
        if t5_precision == "bfloat16":
            dtype = torch.bfloat16
        elif t5_precision == "float16":
            dtype = torch.float16
        elif t5_precision == 'float':
            dtype = torch.float32
        else:
            raise ValueError('not supported t5_precision: {}'.format(t5_precision))
        
        with self.maybe_autocast(dtype=dtype):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                modal_name=modal_name
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        modal_name = "image"

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens_img.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            modal_name=modal_name
        )

        inputs_t5 = self.t5_proj_img(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        self.t5_model.modal_name = modal_name
        self.t5_model.encoder.modal_name = modal_name
        self.t5_model.decoder.modal_name = modal_name
        
        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text
    
    def predict_zs_pc_classification(self, samples):
        # For zero-shot pointcloud classification
        points = samples['pointcloud'].float()
        targets = samples['label']

        with self.maybe_autocast():
            pc_embeds = self.point_encoder(points)
            pc_embeds = self.pc_adapter(pc_embeds)
            pc_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(self.device)
        
        query_tokens = self.query_tokens_pc.expand(pc_embeds.size(0), -1, -1)
        
        if self.enable_pointcloud_qformer:
            qformer = self.Qformer_pc
        else:
            qformer = self.Qformer
        with self.maybe_autocast():
            query_output = qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=pc_atts,
                use_cache=True,
                return_dict=True,
                modal_name="pointcloud"
            )

        pc_feats = F.normalize(self.stage1_proj_pc(query_output.last_hidden_state), dim=-1)     # [batch, #query, 256]

        #logits = 100.0 * pc_feats @ self.point_classifier
        logits = pc_feats @ self.point_classifier                                               # [batch, #query, #category]
        logits = logits.max(dim=1).values                                                       # [batch, #catetory]

        return {"predictions": logits, "targets": targets}
    
    def before_evaluation(self, dataset, task_type, **kwargs):
        from lavis.tasks import Multimodal3DClassification

        if task_type == Multimodal3DClassification:
            modal_name = "pointcloud"
            if self.enable_pointcloud_qformer:
                qformer = self.Qformer_pc
            else:
                qformer = self.Qformer

            # prepare zero-shot point_classifier
            classnames = dataset.classnames
            #templates = dataset.templates
            templates = ["{}."]
            with torch.no_grad():
                zs_weights = []
                for classname in classnames:
                    texts = [
                        t.format(classname) for t in templates]
                    
                    text_tokens = self.tokenizer(
                        texts,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)                                                                       
                    text_output = qformer.bert(
                        text_tokens.input_ids,
                        attention_mask=text_tokens.attention_mask,
                        return_dict=True,
                        modal_name=modal_name)                                                                 
                    # text_output.last_hidden_state: [b, 32, 768]
                    text_feats = self.stage1_proj_text(text_output.last_hidden_state[:, 0, :])      # [#prompt, 256]

                    class_embedding = F.normalize(text_feats, dim=-1).mean(dim=0)                   # (256,)
                    class_embedding /= class_embedding.norm()
                    zs_weights.append(class_embedding)

                self.point_classifier = torch.stack(zs_weights, dim=1).to(self.device)              # (256, #category)
    

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        model = cls(cfg)
        model.load_checkpoint_from_config(cfg)
        return model
