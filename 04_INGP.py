import numpy as np
import torch
import json
import math
import cv2 as cv
from tqdm import trange, tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import argparse
from deepspeed.profiling.flops_profiler import FlopsProfiler

from Data import NeRFSynthetic
from NGP import InstantNGP
from utils import Camera, modify_learning_rate
from Render import render_image, render_image_with_eval_ntk
from EVALUATE_NTK import parse_NTK

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type = str, default = "lego")
parser.add_argument("--config", type = str, default = "base")
parser.add_argument("--max_steps", type = int, default = 25000)
parser.add_argument("--load_snapshot", "--load", type = str, default = "None")
parser.add_argument("--batch", "--batch_size", type = int, default = 1024)
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
    if args.load_snapshot != "None":
        ngp.load_snapshot(args.load_snapshot)
    weight_decay = (
        1e-5 if scene_name in ["materials", "ficus", "drums"] else 1e-6
    )
    hash_grid_params = ngp.hash_g.parameters()
    sig_net_params = ngp.hash_n.parameters()
    col_net_params = ngp.mlp.parameters()
    #print(dict(ngp.named_parameters()).keys())
    #exit()
    #module_names = 
    optimizer_hashg = torch.optim.Adam(
        ngp.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay
    )
    optimizer_hashn = torch.optim.Adam(
        sig_net_params, lr=1e-2, eps=1e-15, weight_decay=weight_decay
    )
    optimizer_mlp = torch.optim.Adam(
        col_net_params, lr=1e-2, eps=1e-15, weight_decay=weight_decay
    )
    # Train Utils
    grad_scaler = torch.amp.GradScaler(2**10)
    # 学习率调度器（每个优化器单独设置）
    scheduler_hashg = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer_hashg, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer_hashg,
            milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
            gamma=0.33
        )
    ])

    scheduler_hashn = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer_hashn, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer_hashn,
            milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
            gamma=0.33
        )
    ])

    scheduler_mlp = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer_mlp, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer_mlp,
            milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
            gamma=0.33
        )
    ])

    ntk_dict_records = []
    modules = [
        "hash_network",
        "rgb_network",
        "hash_encoding"
    ]
    
    prof = FlopsProfiler(ngp)
    frozen_modules = set()
    # Training
    ngp.train()
    ngp.grid.train()
    for step in trange(max_steps + 1):
        def occ_eval_fn(x):
            density = ngp.get_density(x)
            return density * step_length
        ngp.grid.update_every_n_steps(step = step, occ_eval_fn = occ_eval_fn, occ_thre = 1e-2)
        
        if (step % 1000) == 0:
            prof.start_profile()
        pixels, rays_o, rays_d = dataset.sample(batch_size)
        pixels = pixels.cuda()
        color = render_image(
            ngp, ngp.grid, rays_o, rays_d
        )
        loss = torch.nn.functional.smooth_l1_loss(color, pixels)
        optimizer_hashg.zero_grad()
        optimizer_hashn.zero_grad()
        optimizer_mlp.zero_grad()
        grad_scaler.scale(loss).backward()
        if "hash_grid" in frozen_modules:
            for param in ngp.hash_g.parameters():
                param.grad = None
        else:
            optimizer_hashg.step()
            scheduler_hashg.step()
        if "hash_network" in frozen_modules:
            for param in ngp.hash_n.parameters():
                param.grad = None
        else:
            optimizer_hashn.step()
            scheduler_hashn.step()
        if "rgb_network" in frozen_modules:
            for param in ngp.mlp.parameters():
                param.grad = None
        else:
            optimizer_mlp.step()
            scheduler_mlp.step()

        if (step % 5000) == 0:
            prof.stop_profile()
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            prof.print_model_profile(profile_step = step // 1000)
        torch.cuda.empty_cache()
        # Eval
        if step % 1000 == 0:
            with torch.no_grad():
                total_ntk_records = []
                nonzero_cnt = 0
                for i in TEST_IDs:
                    rays_o_total = torch.tensor(camera.rays_o[i: i+1], dtype = torch.float32)
                    rays_d_total = torch.tensor(camera.rays_d[i: i+1], dtype = torch.float32)
                    _, ntk_records = render_image_with_eval_ntk(
                        ngp, ngp.grid, rays_o_total, rays_d_total,
                    )
                    if (len(ntk_records) > 0):
                        total_ntk_records.append(ntk_records)
                        nonzero_cnt += len(ntk_records)
                    if nonzero_cnt > 8:
                        break
                    torch.cuda.empty_cache()
                # Modular Adaptive Training
                ntk_dict: dict = parse_NTK(total_ntk_records)
                if (step > 1000):
                    tqdm.write("=" * 50)
                    for module in modules:
                        start_lambda = ntk_dict_records[0][module]["Max Eigenvalue"]
                        first_Delta_l = np.abs(
                            ntk_dict_records[1][module]["Max Eigenvalue"] - start_lambda
                        )
                        end_lambda = ntk_dict_records[-1][module]["Max Eigenvalue"]

                        Delta_t_l = np.abs(
                            ntk_dict[module]["Max Eigenvalue"] - start_lambda
                        )
                        Delta_bef_l = np.abs(
                            end_lambda - start_lambda
                        )
                        
                        judge_value = np.abs((Delta_t_l - Delta_bef_l) / first_Delta_l)
                        
                        tqdm.write(f"Module {module}, "
                                   f"Max Eigenvalue: {ntk_dict[module]['Max Eigenvalue']:.4e}, "
                                   f"Condition Number: {ntk_dict[module]['Condition Number']:.4e}, "
                                   f"Judge Value: {judge_value:.4e}")
                        
                        if (judge_value < 1e-3):
                            tqdm.write(f"Step {step}, {module} Max Eigenvalue is stable, skip training.")
                            frozen_modules.add(module)
                            # 可选：将对应优化器学习率置为 0（双重保险）
                            if module == "hash_network":
                                for g in optimizer_hashg.param_groups:
                                    g['lr'] = 0.0
                            elif module == "rgb_network":
                                for g in optimizer_mlp.param_groups:
                                    g['lr'] = 0.0
                            elif module == "hash_encoding":
                                for g in optimizer_hashn.param_groups:
                                    g['lr'] = 0.0

                ntk_dict_records.append(ntk_dict)


                torch.save(
                    total_ntk_records, f"./snapshots/{scene_name}_ntk_{step}.pt"
                )
            
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

                cv.imwrite(f"{scene_name}.png", image * 255.)
                psnr = compute_psnr(image, ref)
            tqdm.write(f"Step {step}, PSNR = {round(psnr.item(), 4)}")
            # If All Modules are frozen, stop training
            if len(frozen_modules) == len(modules):
                tqdm.write(f"All modules frozen at step {step}, stopping training.")
                break


        torch.cuda.empty_cache()
        if (step % 5000 == 0) and (step > 0):
            tqdm.write(f"Saving snapshot at step {step}...")
            ngp.save_snapshot(path = f"./snapshots/{scene_name}_{step}.msgpack", load_path = "./snapshots/base.msgpack")