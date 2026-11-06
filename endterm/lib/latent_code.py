"""
This copies the original implementation of anonymize.py but uses the
FM generated layers as the middle layer instead.
"""

import torch
import torch.nn as nn
import os
import os.path as osp

from endterm.training.train_flow import FlowMLP

class FlowLatentCode(nn.Module):
    def __init__(self, latent_code_real, latent_code_flow, img_id, out_code_dir, gan='stylegan2_ffhq1024', latent_space='W+'):
        super(FlowLatentCode, self).__init__()
        self.latent_code_real = latent_code_real
        self.latent_code_flow = latent_code_flow # This is exactly shape (B, 5, 512)
        self.img_id = img_id
        self.gan = gan
        self.latent_space = latent_space
        self.out_code_dir = out_code_dir

        if self.latent_space == 'W+':
            self.layer_start = 3
            self.layer_end = 8

            # Layers 0-2 (Frozen Real)
            self.nontrainable_layers_start = nn.Parameter(data=self.latent_code_real[:, :self.layer_start, :],
                                                          requires_grad=False)
            
            # Layers 3-7 (Trainable Flow Prior) 
            self.trainable_layers = nn.Parameter(data=self.latent_code_flow,
                                                 requires_grad=True)
            
            # Layers 8-17 (Frozen Real)
            self.nontrainable_layers_end = nn.Parameter(data=self.latent_code_real[:, self.layer_end:, :],
                                                        requires_grad=False)

    def save(self):
        if self.latent_space == 'W+':
            torch.save(torch.cat([self.nontrainable_layers_start,
                                  self.trainable_layers,
                                  self.nontrainable_layers_end], dim=1),
                       osp.join(self.out_code_dir, '{}.pt'.format(self.img_id)))

    def forward(self):
        return torch.cat([self.nontrainable_layers_start,
                          self.trainable_layers,
                          self.nontrainable_layers_end], dim=1)