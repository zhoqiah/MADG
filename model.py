"""
Name: MGA
Date: 2023/08/28
Version: 1.0
"""

import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
import os
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
import math
import matplotlib.pyplot as plt
from pre_model import RobertaEncoder
import copy
# from Orthographic_pytorch import Ortho_algorithm_unique,Ortho_algorithm_common
from Vit3 import ViT
from torch.nn import TransformerEncoderLayer, Transformer, MultiheadAttention
from sentence_transformers import SentenceTransformer
from data_process import data_process
from transformers import ViTModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict

from functools import partial
# from models.vit import VisionTransformer
# from models.xbert import BertConfig, BertForMaskedLM

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, text=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.text = text

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, text=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.text = text


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = './weights/'

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)
            self.model = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = False  # True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        # input = input.texts
        output = self.model(input, attention_mask=attention_mask)
        return output


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        self.text_model = TextModel(opt)
        self.vit = ViTModel.from_pretrained('facebook/deit-base-patch16-224')
        self.text_config = copy.deepcopy(self.text_model.get_config())
        self.image_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers

        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)
        self.image_encoder = RobertaEncoder(self.image_config)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            # nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            nn.Linear(2048, opt.tran_dim),
            ActivateFun(opt)
        )  # 2048
        self.image_cls_change = nn.Sequential(
            # nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            nn.Linear(2048, opt.tran_dim),
            ActivateFun(opt)
        )  # 2048
        # self.encoder_layer = TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer = Transformer(nhead=1, num_encoder_layers=1, num_decoder_layers=1, d_model=768, dim_feedforward=128)  # 4,1,1,768,128;

        self.multiheadattention = MultiheadAttention(768, 16, dropout=0.1)
        # self.multiheadattention = layers.MultiHeadAttention(head, dim, dim, dim, dropout=dropout)
        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim//64, dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=opt.tran_num_layers)
        # self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_transformer = SentenceTransformer(
                                            'microsoft/mpnet-base',
                                            cache_folder="./weights/sentence_transformers")


        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)  # 此处需要注意
        )


    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask, text):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        # text_encoder = self.text_model(text_inputs.texts, attention_mask=bert_attention_mask.bert_attention_mask)
        # sentence = self.sentence_transformer.encode(text, convert_to_tensor=True)
        # sentence = sentence.unsqueeze(dim=0).cuda()
        # text_cls = text_encoder.pooler_output

        # print('text: ', text)
        # exit()

        text_encoder = text_encoder.last_hidden_state
        text_init = self.text_change(text_encoder)
        # vit模型
        image_feature = self.vit(image_inputs)
        image_init = image_feature.last_hidden_state
        # image_encoder, image_cls = self.v(image_inputs)
        for param in self.vit.parameters():
            param.requires_grad = False

        # image_init = image_init.permute(1, 0, 2).contiguous()
        # text_init = text_init.permute(1, 0, 2).contiguous()
        # text_image_cat = self.transformer(image_init, text_init)
        # # text_image_cat = self.transformer(sentence, text_image_cat)
        # text_image_transformer = text_image_cat.permute(1, 2, 0).contiguous()
        # concat
        text_image_transformer = torch.cat((text_init, image_init), dim=1)
        text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()


        if self.fuse_type == 'max':
            text_image_output = torch.max(text_image_transformer, dim=2)[0]
        elif self.fuse_type == 'att':
            text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()

            text_image_mask = text_image_mask.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:text_image_output.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention(text_image_output)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

            text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)
        elif self.fuse_type == 'ave':
            text_image_length = text_image_transformer.size(2)
            text_image_output = torch.sum(text_image_transformer, dim=2) / text_image_length
        else:
            raise Exception('fuse_type设定错误')
        cap_length = np.array([text_image_length]*len(text))

        return text_image_output, image_init, text_init, cap_length


class CLModel(nn.Module):
    def __init__(self, opt, temp=0.07):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)
        # self.temperature = opt.temperature
        self.set_cuda = opt.cuda

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.augment_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)  # 2 3 6
        )

        # ITCLoss
        # self.vision_proj = nn.Linear(vision_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)
        # bert_config = BertConfig.from_json_file(bert_config_path)
        # create momentum models
        # self.visual_encoder_m = VisionTransformer(
            # img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            # mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        # self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * temp)
        # create the queue
        self.queue_size = 65536
        self.register_buffer("image_queue", torch.randn(opt.tran_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(opt.tran_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


    def forward(self, data_orgin, data_augment = None, labels=None, target_labels=None, text=None, alpha = .5):
        orgin_res, image_init, text_init, text_length = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                     data_orgin.images, data_orgin.text_image_mask,text)

        # """grad_cam"""
        # orgin_res, image_init, text_init, text_length = self.fuse_model(data_orgin.texts,
        #                                                                 data_orgin.bert_attention_mask,
        #                                                                 data_orgin.images, data_orgin.text_image_mask,
        #                                                                 data_orgin.text)
        output = self.output_classify(orgin_res)

        # ITCLoss
        image_feat = F.normalize(image_init[:, 0, :], dim=-1)
        text_feat = F.normalize(text_init[:, 0, :], dim=-1)
        # get momentum features
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
            # self._momentum_update()
            # image_embeds_m = self.visual_encoder_m(data_orgin.images)
            # image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)
            image_feat_m = image_feat
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)

            # text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,
                                                # return_dict = True, mode = 'text')
            # text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
            text_feat_m = text_feat
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(data_orgin.images.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        return output, image_init, text_init, text_length, loss_ita

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)
        image_feats = image_feat
        text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

class CLModel_bak(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)
        # self.temperature = opt.temperature
        self.set_cuda = opt.cuda

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.augment_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)  # 2 3 6
        )


    def forward(self, data_orgin, data_augment = None, labels=None, target_labels=None, text=None):
        orgin_res, image_init, text_init, text_length = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                     data_orgin.images, data_orgin.text_image_mask,text)

        # """grad_cam"""
        # orgin_res, image_init, text_init, text_length = self.fuse_model(data_orgin.texts,
        #                                                                 data_orgin.bert_attention_mask,
        #                                                                 data_orgin.images, data_orgin.text_image_mask,
        #                                                                 data_orgin.text)
        output = self.output_classify(orgin_res)
        return output, image_init, text_init, text_length


class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = FuseModel(opt)

    def forward(self, texts, bert_attention_mask, images, text_image_mask,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images, text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])
