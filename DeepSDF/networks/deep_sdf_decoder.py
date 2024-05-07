#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 66] + dims + [1]#########################################

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 66 #########################################################
                    print(out_dim)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -66:]###########################################

        if input.shape[1] > 66 and self.latent_dropout:##########################
            latent_vecs = input[:, :-66]##########################
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)

            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)

            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

# class Decoder(torch.nn.Module):
#     def __init__(
#         self,
#         latent_size,
#         dims,
#         dropout=None,
#         dropout_prob=0.0,
#         norm_layers=(),
#         latent_in=(),
#         weight_norm=False,
#         xyz_in_all=None,
#         use_tanh=False,
#         latent_dropout=False,
#     ):
#         """
#         dim[0]: input dim
#         dim[1:-1]: hidden dims
#         dim[-1]: out dim

#         assume len(dims) >= 3
#         """
#         super().__init__()

#         dims = [latent_size + 66] + dims + [1]

#         self.num_layers = len(dims)
#         self.norm_layers = norm_layers
#         self.latent_in = latent_in
#         self.latent_dropout = latent_dropout
#         if self.latent_dropout:
#             self.lat_dp = nn.Dropout(0.2)

#         self.xyz_in_all = xyz_in_all
#         self.weight_norm = weight_norm       

#         self.layers = torch.nn.ModuleList()
#         for ii in range(len(dims)-2):
#             self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))

#         self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
#         self.relu = torch.nn.ReLU()

#     def forward(self, x):
#         for ii in range(len(self.layers)):
#             x = self.layers[ii](x)
#             x = self.relu(x)
#         return self.layer_output(x)
    
# class LipschitzLinear(torch.nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
#         self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
#         self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
#         self.softplus = torch.nn.Softplus()
#         self.initialize_parameters()

#     def initialize_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)

#         # compute lipschitz constant of initial weight to initialize self.c
#         W = self.weight.data
#         W_abs_row_sum = torch.abs(W).sum(1)
#         self.c.data = W_abs_row_sum.max() # just a rough initialization

#     def get_lipschitz_constant(self):
#         return self.softplus(self.c)

#     def forward(self, input):
#         lipc = self.softplus(self.c)
#         scale = lipc / torch.abs(self.weight).sum(1)
#         scale = torch.clamp(scale, max=1.0)
#         return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)



# class Decoder(torch.nn.Module):
#     def __init__(
#         self,
#         latent_size,
#         dims,
#         dropout=None,
#         dropout_prob=0.0,
#         norm_layers=(),
#         latent_in=(),
#         weight_norm=False,
#         xyz_in_all=None,
#         use_tanh=False,
#         latent_dropout=False,
#     ):
#         """
#         dim[0]: input dim
#         dim[1:-1]: hidden dims
#         dim[-1]: out dim

#         assume len(dims) >= 3
#         """
#         super().__init__()

#         dims = [latent_size + 66] + dims + [1]

#         self.num_layers = len(dims)
#         self.norm_layers = norm_layers
#         self.latent_in = latent_in
#         self.latent_dropout = latent_dropout
#         if self.latent_dropout:
#             self.lat_dp = nn.Dropout(0.2)

#         self.xyz_in_all = xyz_in_all
#         self.weight_norm = weight_norm       

#         self.layers = torch.nn.ModuleList()
#         for ii in range(len(dims)-2):
#             if ii + 1 in latent_in:
#                 out_dim = dims[ii + 1] - dims[0]
#             else:
#                 out_dim = dims[ii + 1]
#                 if self.xyz_in_all and ii != self.num_layers - 2:
#                     out_dim -= 60
            
#             if weight_norm and ii in self.norm_layers:
#                 self.layers.append(nn.utils.weight_norm(LipschitzLinear(dims[ii], out_dim)))
#             else:
#                 self.layers.append(LipschitzLinear(dims[ii], out_dim))

#         self.layer_output = LipschitzLinear(dims[-2], dims[-1])
#         self.relu = torch.nn.ReLU()

#         self.dropout_prob = dropout_prob
#         self.dropout = dropout
#         self.th = nn.Tanh()

#     def get_lipschitz_loss(self):
#         loss_lipc = 1.0
#         for ii in range(len(self.layers)):
#             loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
#         loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
#         return loss_lipc

#     def forward(self, x):
#         input = x
#         xyz = input[:, -60:]

#         for ii in range(len(self.layers)):

#             if ii in self.latent_in:
#                 x = torch.cat([x, input], 1)
#             elif ii != 0 and self.xyz_in_all:
#                 x = torch.cat([x, xyz], 1)

#             x = self.layers[ii](x)
#             x = self.relu(x)
#             if self.dropout is not None and ii in self.dropout:
#                 x = F.dropout(x, p=self.dropout_prob, training=self.training)

#         return self.layer_output(x)