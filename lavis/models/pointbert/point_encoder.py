import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from lavis.models.pointbert.dvae import Group
from lavis.models.pointbert.dvae import Encoder
#from lavis.models.pointbert.logger import print_log

from lavis.models.pointbert.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, trans_dim=384, depth=12, drop_path_rate=0.1,
                 cls_dim=40, num_heads=6, group_size=32, num_group=512,
                 encoder_dims=256,
                 #out_dims=784,
                 with_color=False,
                 pretrain_weight="./data/initialize_models/point_bert_pretrained.pt"):
        super().__init__()
        self.with_color = with_color
        self.trans_dim = trans_dim                                                                  # 384
        self.depth = depth                                                                          # 12
        self.drop_path_rate = drop_path_rate                                                        # 0.1
        self.cls_dim = cls_dim                                                                      # 40
        self.num_heads = num_heads                                                                  # 6

        self.group_size = group_size                                                                # 32
        self.num_group = num_group                                                                  # 512
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)            # <class 'lavis.models.pointbert.dvae.Group'>
        # define the encoder
        self.encoder_dims = encoder_dims                                                            # 256
        self.encoder = Encoder(encoder_channel=self.encoder_dims)                                   # <class 'lavis.models.pointbert.dvae.Encoder'>
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        # add by jianchong
        #self.adapter_dim = nn.Linear(self.trans_dim * 2, out_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
            )
        
        if with_color:
            logging.info('point encoder: with_color is True')
            from .dvae import EncoderWithColor
            self.encoder_color = EncoderWithColor(encoder_channel=self.encoder_dims)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        assert os.path.isfile(pretrain_weight)
        self.load_model_from_ckpt(pretrain_weight)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):        
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]
        
        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(get_missing_parameters_message(incompatible.missing_keys))
            #print_log('missing_keys', logger='Transformer')
            #print_log(
            #    get_missing_parameters_message(incompatible.missing_keys),
            #    logger='Transformer'
            #)
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(get_unexpected_parameters_message(incompatible.unexpected_keys))
            #print_log('unexpected_keys', logger='Transformer')
            #print_log(
            #    get_unexpected_parameters_message(incompatible.unexpected_keys),
            #    logger='Transformer'
            #)

        #print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        logging.info('Successful Loading the ckpt from {}'.format(bert_ckpt_path))

    def forward(self, pts, return_dense_features=True, return_center=False):
        # pts: [batch, 8192, 3]

        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        #   neighborhood: [batch, num_group, group_size, 3]
        #   center: [batch, num_group, 3]

        # encoder the input cloud blocks
        if self.with_color:
            assert neighborhood.size(-1) == 6
            group_input_tokens = self.encoder_color(neighborhood)
        else:
            assert neighborhood.size(-1) == 3
            group_input_tokens = self.encoder(neighborhood)                             # [batch, num_group, 256]
        group_input_tokens = self.reduce_dim(group_input_tokens)                    # [batch, num_group, 384]

        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)      # [batch, 1, 384]
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)           # [batch, 1, 384]

        # add pos embedding
        pos = self.pos_embed(center)                                                # [batch, num_group, 384]

        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)                      # [batch, 1+num_group, 384]
        pos = torch.cat((cls_pos, pos), dim=1)                                      # [batch, 1+num_group, 384]
        # transformer
        x = self.blocks(x, pos)                                                     # [batch, 1+num_group, 384]
        x = self.norm(x)
        
        if return_dense_features:
            if return_center:
                return x, center
            else:
                return x
        else:
            concat_f = torch.cat([x[:, 0], x[:, 1:].max(dim=1)[0]], dim=-1)         # [batch, 768]
            concat_f = concat_f.unsqueeze(1)
            #concat_f = self.adapter_dim(concat_f)

            if return_center:
                return concat_f, center
            else:
                return concat_f
            