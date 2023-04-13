#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

try:
    from layers import *
except:
    from src.layers import *


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x




class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(x)
        return x



def cosine_sim(im, s):
    return im.mm(s.t())


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)
    return X


class AttrEncoder(nn.Module):
    def __init__(self, args, attr_list):
        super().__init__()
        self.args = args
        self.attr_embed = nn.Embedding.from_pretrained(torch.FloatTensor(attr_list))
        print('attr_list111', torch.FloatTensor(attr_list).shape)
        self.fc1 = nn.Linear(768, self.args.attr_dim)
        self.fc2 = nn.Linear(200, self.args.attr_dim)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)

    def forward(self, e_a, e_v, mask, l, i):
        e_a = self.fc1(self.attr_embed(e_a))
        e_v = torch.sigmoid(e_v.unsqueeze(-1)).repeat(1, 1, 100)
        e = self.fc2(torch.cat([e_a, e_v], dim=2))
        e = e.repeat(6,1,1)
        e = e[:27793,:,:]
        mask = mask.repeat(6, 1)
        mask = mask[:27793,:]
        alpha = F.softmax(torch.sum(e * i.unsqueeze(1), dim=-1) * mask, dim=1)
        e = torch.sum(alpha.unsqueeze(2) * e, dim=1)
        # e = torch.sum(e * i.unsqueeze(0).unsqueeze(0), dim=(1, 2))
        return e


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        # joint_emb = torch.sum(torch.stack(embs, dim=1), dim=1)
        return joint_emb


class MultiModalEncoder(nn.Module):
    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_list=None):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        # nn.init.xavier_normal_(self.entity_emb.weight.data)
        self.entity_emb.requires_grad = True

        self.rel_fc = nn.Linear(1000, attr_dim)
        # 新增
        # nn.init.xavier_normal_(self.rel_fc.weight.data)
        self.att_fc = nn.Linear(1000, attr_dim)
        # nn.init.xavier_normal_(self.att_fc.weight.data)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        # number attr encoder
        self.attr_encoder = AttrEncoder(self.args, attr_list)
        self.fc_a = nn.Linear(1, attr_dim)
        self.fc_a1 = nn.Linear(attr_dim, attr_dim)
        nn.init.xavier_normal_(self.fc_a.weight.data)
        nn.init.xavier_normal_(self.fc_a1.weight.data)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(self.n_units[2], self.n_units[2], self.n_units[2], dropout)


        self.fusion = MultiModalFusion(modal_num=self.args.inner_view_num,
                                       with_weight=self.args.with_weight)

    def forward(self,
                input_idx,
                adj,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None,
                entity_num_attr=None,
                entity_num_val=None,
                mask_num=None,
                l1_num=None,
                attr_encoder=None):

        # 结构编码训练
        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
            # print('rel_emb大小', rel_emb.shape)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.w_name:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        att_num_emd = self.attr_encoder(entity_num_attr, entity_num_val, mask_num, l1_num, img_emb)

        if self.use_project_head:
            gph_emb = self.gph_pro(gph_emb)
            img_emb = self.img_pro(img_emb)
            rel_emb = self.rel_pro(rel_emb)
            att_emb = self.att_pro(att_emb)
            pass

        joint_emb = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb, att_num_emd])

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, att_num_emd, joint_emb
