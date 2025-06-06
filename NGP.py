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

class InstantNGP(torch.nn.Module):
    def __init__(self, config, use_self_model: bool = False):
        super().__init__()
        self.config = config
        
        # Consts
        self.near = 0.6
        self.far = 2.0
        self.steps = 1024
        self.step_length = math.sqrt(3) / self.steps
        
        # Initialize models
        if use_self_model:
            hashgrid = Hash.HashEncoding(
                n_input_dims = 3,
                encoding_config = config["encoding"],
            ).to("cuda")
            shenc = SH.SHEncoding(
                n_input_dims = 3,
                encoding_config = config["dir_encoding"],
            ).to("cuda")
        else:
            hashgrid = tcnn.Encoding(
                n_input_dims = 3,
                encoding_config = config["encoding"],
            ).to("cuda")

            shenc = tcnn.Encoding(
                n_input_dims = 3,
                encoding_config = config["dir_encoding"],
            ).to("cuda")

        sig_net = Network.MLP(
            n_input_dims = 32,
            n_output_dims = 16,
            network_config = config["network"]
        ).to("cuda")
        rgb_net = Network.MLP(
            n_input_dims = 32,
            n_output_dims = 3,
            network_config = config["rgb_network"]
        ).to("cuda")

        self.hash_g: torch.nn.Module = hashgrid
        self.hash_n: torch.nn.Module = sig_net
        self.sh: torch.nn.Module = shenc
        self.mlp: torch.nn.Module = rgb_net
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


    def load_snapshot(self, path: str):
        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
            snapshot = next(unpacker)
        self.snapshot = snapshot
        params_binary = torch.tensor(
            np.frombuffer(snapshot["snapshot"]["params_binary"], 
                        dtype = np.float16, offset = 0)#.astype(np.float32)
            , dtype = torch.float32)
        # Params for Hash Encoding Network
        ## Network Params Size: 32 * 64 + 64 * 16 = 3072
        params_hashnet = params_binary[:(32 * 64 + 64 * 16)]
        params_binary = params_binary[(32 * 64 + 64 * 16):]
        # Params for RGB Network
        ## Network Params Size: 32 * 64 + 64 * 64 + 64 * 16 = 7168
        params_rgbnet = params_binary[:(32 * 64 + 64 * 64 + 64 * 16)]
        params_binary = params_binary[(32 * 64 + 64 * 64 + 64 * 16):]
        # Params for Hash Encoding Grid
        params_hashgrid = params_binary
        # Params of Density Grid
        grid_raw = torch.tensor(
            np.frombuffer(
                snapshot["snapshot"]["density_grid_binary"], dtype=np.float16).astype(np.float32),
            dtype = torch.float32
            )
        grid = torch.zeros([128 * 128 * 128], dtype = torch.float32)

        x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
        grid[x * 128 * 128 + y * 128 + z] = grid_raw
        grid_3d = torch.reshape(grid > 0.01, [1, 128, 128, 128]).type(torch.bool)
        
        self.hash_g.load_states(params_hashgrid)
        self.hash_n.load_states(params_hashnet)
        self.mlp.load_states(params_rgbnet)
        self.grid.load_state_dict({
            "resolution": torch.tensor([128, 128, 128], dtype = torch.int32),
            "aabbs": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "occs": grid,
            "binaries": grid_3d
        })
    
    def save_snapshot(self, path: str, load_path: str | None = None):
        if load_path is not None:
            with open(load_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
                snapshot = next(unpacker)
        # Parameters
        hash_g_params_raw = []
        for k, v in self.hash_g.state_dict().items():
            hash_g_params_raw.append(v.reshape([-1,]))
        params_hashgrid = torch.cat(hash_g_params_raw)
        hash_n_params_raw = []
        for k, v in self.hash_n.state_dict().items():
            hash_n_params_raw.append(v.reshape([-1,]))
        params_hashnet = torch.cat(hash_n_params_raw)
        params_hash = torch.cat([params_hashnet, params_hashgrid], dim = -1).clone().cpu().detach()
        rgbnet_params_raw = []
        for k, v in self.mlp.state_dict().items():
            rgbnet_params_raw.append(v.reshape([-1,]))
        rgbnet_params_raw.append(torch.zeros([64 * 13], device = "cuda"))
        params_rgbnet = torch.cat(rgbnet_params_raw).clone().cpu().detach()
        params_binary = torch.cat([
            params_hash[:(32 * 64 + 64 * 16)],
            params_rgbnet,
            params_hash[(32 * 64 + 64 * 16):]
        ]).numpy()
        snapshot["snapshot"]["params_binary"] = np.float16(params_binary).tobytes()
        # Density Grids
        density_grid: torch.Tensor = self.grid.state_dict()["occs"].clone().cpu().detach().type(torch.float16)
        grid_morton = torch.zeros(128 ** 3, dtype = torch.float16)
        indexs = torch.arange(0, 128**3, 1)
        grid_morton[morton_naive(indexs // (128 * 128), (indexs % (128 * 128)) // 128, indexs % 128)] = density_grid
        snapshot["snapshot"]["density_grid_binary"] = grid_morton.detach().numpy().tobytes()
        
        # HyperParameters
        snapshot['encoding']['log2_hashmap_size'] = self.config["encoding"]["log2_hashmap_size"]
        snapshot['encoding']['n_levels'] = self.config["encoding"]["n_levels"]
        snapshot['encoding']['n_features_per_level'] = self.config["encoding"]["n_features_per_level"]
        snapshot['encoding']['base_resolution'] = self.config["encoding"]["base_resolution"]
        with open(path, 'wb') as f:
            f.write(msgpack.packb(snapshot))
    
    def get_alpha(self, x: torch.Tensor):
        hash_features_raw = self.hash_g(x)
        hash_features_raw = hash_features_raw.type(torch.float32)
        hash_features = self.hash_n(hash_features_raw)
        alphas_raw = hash_features[..., 0]
        alphas = (1. - torch.exp(-torch.exp(alphas_raw.type(torch.float32)) * self.step_length))
        return alphas

    def get_density(self, x: torch.Tensor):
        hash_features_raw = self.hash_g(x)
        hash_features_raw = hash_features_raw.type(torch.float32)
        hash_features = self.hash_n(hash_features_raw)
        alphas_raw = hash_features[..., 0]
        density = torch.exp(alphas_raw - 1)
        return density

    def get_rgb(self, x: torch.Tensor, dir: torch.Tensor):
        hash_features_raw = self.hash_g(x)
        hash_features_raw = hash_features_raw.type(torch.float32)
        hash_features = self.hash_n(hash_features_raw)
        sh_features = self.sh((dir + 1) / 2)
        sh_features = sh_features.type(torch.float32)      
        features = torch.concat([hash_features, sh_features], dim = -1)
        rgbs_raw = self.mlp(features)
        rgbs = torch.sigmoid(rgbs_raw)
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
        func_hash, params_hash = GetFuncParams(self.hash_g, model_type="hash")
        ntk_hash = Evaluate_NTK(func_hash, params_hash, position, position, compute='mNTK')
        for i in range(16):
            # 定义子函数，捕获当前的i值（避免延迟绑定问题）
            def func_sub(params, x, group_idx=i):
                full_output = func_hash(params, x)
                return full_output[2*group_idx : 2*group_idx + 2]
            # 计算子模型的NTK
            ntk_sub = Evaluate_NTK(func_sub, params_hash, position, position, compute='mNTK')
            #print(f"Group {i} NTK shape: {ntk_sub.shape}")
            # 保存到字典中
            ntks[f"hash_group_{i}"] = ntk_sub

        ntks["hash_encoding"] = ntk_hash#.cpu().detach()

        #func_hash_lev, params_hash_lev = GetFuncParams(
        #    self.hash_g.forward_single_level, model_type="hash"
        #)
        #ntk_hash_lev = Evaluate_NTK(
        #    func_hash_lev, params_hash_lev, position, position, compute='mNTK'
        #)
        #ntks["hash_encoding_single_level"] = ntk_hash_lev#.cpu().detach()

        # Hash Network
        sigma_features = self.hash_n(hash_features_raw)
        func_sigma, params_sigma = GetFuncParams(self.hash_n, model_type="linear")
        ntk_sigma = Evaluate_NTK(func_sigma, params_sigma, hash_features_raw, hash_features_raw, compute='mNTK')
        ntks["hash_network"] = ntk_sigma#.cpu().detach()
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
        ntks["rgb_network"] = ntk_rgb#.cpu().detach()
        rgbs = torch.sigmoid(rgbs_raw)

        return rgbs, alphas, ntks