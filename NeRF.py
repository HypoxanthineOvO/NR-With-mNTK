import numpy as np
import torch

import Modules.Hash as Hash
import Modules.SH as SH
import Modules.Networks as Network

import tinycudann as tcnn

import msgpack
import nerfacc
import math
from utils import inv_morton_naive, morton_naive

from NTK import GetFuncParams, Evaluate_NTK

class NeRF(torch.nn.Module):
    def __init__(self, 
        config: dict,
        use_self_model: bool = False
    ):
        super().__init__()
        self.config = config
        
        # Consts
        self.near = 0.6
        self.far = 2.0
        self.steps = 1024
        self.step_length = math.sqrt(3) / self.steps
        
        # Initialize models
        if use_self_model:
            raise NotImplementedError("Self model is not implemented for NeRF")
        else:
            pos_enc = tcnn.Encoding(
                n_input_dims = 3,
                encoding_config = config["nerf_encoding"]
            )
            dir_enc = tcnn.Encoding(
                n_input_dims = 3,
                encoding_config = config["dir_encoding"]
            )

        base_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(63, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        base_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(319, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        sigma_mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 1)
        )
        color_mlp = torch.nn.Sequential(
            torch.nn.Linear(283, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)
        )

        self.pos_enc: torch.nn.Module = pos_enc
        self.dir_enc: torch.nn.Module = dir_enc

        self.base_mlp_1: torch.nn.Module = base_mlp_1
        self.base_mlp_2: torch.nn.Module = base_mlp_2
        self.sigma_mlp: torch.nn.Module = sigma_mlp
        self.color_mlp: torch.nn.Module = color_mlp

        
        self.grid: torch.nn.Module = nerfacc.OccGridEstimator(
            roi_aabb = [0, 0, 0, 1, 1, 1],
            resolution = 128, levels = 1
        ).to("cuda")
        self.snapshot = None
        
        # Initialize Density Grid
        grid_1d = torch.abs(torch.randn(128 ** 3, dtype = torch.float32)) / 100
        
        grid_3d = torch.reshape((grid_1d > 0.01), [1, 128, 128, 128]).type(torch.bool)
        init_params_grid = {
            "resolution": torch.tensor([128, 128, 128], dtype = torch.int32),
            "aabbs": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "occs": grid_1d,
            "binaries": grid_3d
        }
        self.grid.load_state_dict(init_params_grid)
    

    def evaluate_sigma(self, x: torch.Tensor):
        pos_enc_features = self.pos_enc(x)
        raw_features = self.base_mlp_1(pos_enc_features)
        base_sigma_features = self.base_mlp_2(
            torch.cat([raw_features, pos_enc_features], dim=-1)
        )
        sigma_raw = self.sigma_mlp(base_sigma_features)
        return sigma_raw
    
    def get_alpha(self, x: torch.Tensor):
        alpha = (1. - torch.exp(-torch.relu(self.evaluate_sigma(x).type(torch.float32)) * self.step_length))
        return alpha

    def get_density(self, x: torch.Tensor):
        sigma = torch.exp(self.evaluate_sigma(x).type(torch.float32) - 1)
        return sigma

    def get_rgb(self, x: torch.Tensor, dir: torch.Tensor):
        pos_enc_features = self.pos_enc(x)
        raw_features = self.base_mlp_1(pos_enc_features)
        base_sigma_features = self.base_mlp_2(
            torch.cat([raw_features, pos_enc_features], dim=-1)
        )
        sigma_raw = self.sigma_mlp(base_sigma_features)
        dir_enc_features = self.dir_enc((dir + 1) / 2)
        features = torch.cat([base_sigma_features, dir_enc_features, sigma_raw.unsqueeze(-1)], dim=-1)
        rgb_raw = self.color_mlp(features)
        rgbs = torch.sigmoid(rgb_raw)
        return rgbs

    def forward(self, position, direction):
        alphas = self.get_alpha(position)
        rgbs = self.get_rgb(position, direction)
        return rgbs, alphas
    
    def forward_with_evaluate_ntk(self, position: torch.Tensor, direction: torch.Tensor):
        ntks = {}
        # Hash Encoding
        hash_features_raw = self.hash_g(position)
        hash_features_raw = hash_features_raw.type(torch.float32)
        #func_hash, params_hash = GetFuncParams(self.hash_g, model_type="linear")
        #ntk_hash = Evaluate_NTK(func_hash, params_hash, position, position, compute='mNTK')
        #ntks["hash_encoding"] = ntk_hash

        # Hash Network
        sigma_features = self.hash_n(hash_features_raw)
        func_sigma, params_sigma = GetFuncParams(self.hash_n, model_type="linear")
        ntk_sigma = Evaluate_NTK(func_sigma, params_sigma, hash_features_raw, hash_features_raw, compute='mNTK')
        ntks["hash_network"] = ntk_sigma
        alphas_raw = sigma_features[..., 0]
        alphas = (1. - torch.exp(-torch.exp(alphas_raw.type(torch.float32)) * self.step_length))
        # SH Encoding
        sh_features = self.sh((direction + 1) / 2)   
        sh_features = sh_features.type(torch.float32)     
        features = torch.concat([sigma_features, sh_features], dim = -1)
        # RGB Network
        rgbs_raw = self.mlp(features)
        func_rgb, params_rgb = GetFuncParams(self.mlp, model_type="linear")
        ntk_rgb = Evaluate_NTK(func_rgb, params_rgb, features, features, compute='mNTK')
        ntks["rgb_network"] = ntk_rgb
        rgbs = torch.sigmoid(rgbs_raw)

        return rgbs, alphas, ntks