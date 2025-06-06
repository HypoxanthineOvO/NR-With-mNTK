import numpy as np
import torch
import json
import math
import cv2 as cv
from tqdm import trange, tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import argparse
from deepspeed.profiling.flops_profiler import FlopsProfiler

from Data import NeRFSynthetic, load_density_grid
from NGP import InstantNGP
from utils import Camera, modify_learning_rate
from Render import render_image, render_image_with_eval_ntk
from EVALUATE_NTK import parse_NTK

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type = str, default = "lego")
parser.add_argument("--config", type = str, default = "base")
parser.add_argument("--max_steps", type = int, default = 10000)
parser.add_argument("--load_snapshot", "--load", type = str, default = "None")
parser.add_argument("--batch", "--batch_size", type = int, default = 1024*8)
parser.add_argument("--near_plane", "--near", type = float, default = 0.6)
parser.add_argument("--far_plane", "--far", type = float, default = 2.0)
parser.add_argument("--ray_marching_steps", type = int, default = 1024)

if __name__ == "__main__":
    args = parser.parse_args()
    config_path = f"./configs/{args.config}.json"
    scene_name = args.scene
    max_steps = args.max_steps
    batch_size = args.batch
    near = args.near_plane
    far = args.far_plane
    ngp_steps = args.ray_marching_steps
    step_length = math.sqrt(3) / ngp_steps

    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Evaluate Parameters
    with open(f"./data/nerf_synthetic/{scene_name}/transforms_test.json", "r") as f:
        meta = json.load(f)
    m_Camera_Angle_X = float(meta["camera_angle_x"])
    m_C2W = np.array(meta["frames"][0]["transform_matrix"]).reshape(4, 4)
    camera = Camera((800, 800), m_Camera_Angle_X, m_C2W)
    ref_raw = cv.imread(f"./data/nerf_synthetic/{scene_name}/test/r_0.png", cv.IMREAD_UNCHANGED) / 255.
    ref_raw = ref_raw[..., :3] * ref_raw[..., 3:]
    ref = np.array(ref_raw, dtype=np.float32)
    
    TEST_IDs = np.arange(0, 800 * 800, 100 * 100, dtype = np.int32)
    
    # Datasets
    dataset = NeRFSynthetic(f"./data/nerf_synthetic/{scene_name}")
    
    # Initialize models
    ngp: InstantNGP = InstantNGP(config, True).to("cuda")


    prof = FlopsProfiler(ngp)

    # 将原 trange 包装为可动态更新描述的进度条对象
    steps = {
        "hash_g": 25000,
        "hash_n": 25000,
        "mlp": 17000,
    }
    # Hash_g
    ngp.load_snapshot(f"./snapshots/00_Baseline/ngp_step_{steps["hash_g"]}.msgpack", modules = ["hash_g"])
    ngp.load_snapshot(f"./snapshots/00_Baseline/ngp_step_{steps["hash_n"]}.msgpack", modules = ["hash_n"])
    ngp.load_snapshot(f"./snapshots/00_Baseline/ngp_step_{steps["mlp"]}.msgpack", modules = ["mlp"])

    prof.start_profile()
    
    # 数据采样与渲染计算
    pixels, rays_o, rays_d = dataset.sample(batch_size)
    pixels = pixels.cuda()
    color = render_image(ngp, ngp.grid, rays_o, rays_d)
    loss = torch.nn.functional.smooth_l1_loss(color, pixels)


    prof.stop_profile()
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    #prof.print_model_profile(profile_step = step // 1000, output_file = f"profile_{step}.txt")

    with torch.no_grad():
        total_color = np.zeros([800 * 800, 3], dtype = np.float32)
        val_batch = 10000
        for i in range(0, 800*800, val_batch):
            rays_o_total = torch.tensor(camera.rays_o[i: i+val_batch], dtype = torch.float32)
            rays_d_total = torch.tensor(camera.rays_d[i: i+val_batch], dtype = torch.float32)
            color = render_image(
                ngp, ngp.grid, rays_o_total, rays_d_total,
            ).cpu().detach().numpy()
            total_color[i: i+val_batch] = color
            torch.cuda.empty_cache()
        image = np.clip(total_color[..., [2, 1, 0]].reshape(800, 800, 3), 0, 1)

        cv.imwrite(f"{scene_name}_eval.png", image * 255.)
        psnr = compute_psnr(image, ref)
        print(f"PSNR: {psnr:.4f}")

