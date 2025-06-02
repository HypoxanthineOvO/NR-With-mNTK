import torch
import nerfacc
import math

def render_image(
    ngp_model: torch.nn.Module,
    grid: nerfacc.OccGridEstimator,
    rays_o_total: torch.Tensor, rays_d_total: torch.Tensor,
    near: float = 0.6, far: float = 2.0, step_size: float = math.sqrt(3) / 1024,
    batch_size = 10000
):
    num_pixels = rays_o_total.shape[0]
    rays_o = rays_o_total.cuda()
    rays_d = rays_d_total.cuda()
    
    def rgb_alpha_fn(t_starts, t_ends, ray_indices):
        origins = rays_o[ray_indices]
        directions = rays_d[ray_indices]
        ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
        positions = origins + directions * ts
        
        rgbs, alphas = ngp_model(positions, directions)
        return rgbs, alphas    

    
    ray_indices, t_starts, t_ends = grid.sampling(
        rays_o, rays_d, near_plane = near, far_plane = far, 
        #alpha_fn = alpha_fn,
        render_step_size = step_size
    )
    if(ray_indices.shape[0] <= 0):
        return torch.zeros([num_pixels, 3]).cuda()

    color, opacity, depth, extras = nerfacc.rendering(
        t_starts, t_ends, ray_indices, 
        n_rays = num_pixels, rgb_alpha_fn = rgb_alpha_fn
    )
    return color



def render_image_with_eval_ntk(
    ngp_model: torch.nn.Module,
    grid: nerfacc.OccGridEstimator,
    rays_o_total: torch.Tensor, rays_d_total: torch.Tensor,
    near: float = 0.6, far: float = 2.0, step_size: float = math.sqrt(3) / 1024,
    batch_size = 10000
):
    num_pixels = rays_o_total.shape[0]
    rays_o = rays_o_total.cuda()
    rays_d = rays_d_total.cuda()
    
    ntk_records = []

    def rgb_alpha_fn(t_starts, t_ends, ray_indices):
        origins = rays_o[ray_indices]
        directions = rays_d[ray_indices]
        ts = torch.reshape((t_starts + t_ends) / 2.0, (-1, 1))
        positions = origins + directions * ts
        
        rgbs, alphas, ntks = ngp_model.forward_with_evaluate_ntk(positions, directions)
        ntk_records.append(ntks)
        return rgbs, alphas    

    
    ray_indices, t_starts, t_ends = grid.sampling(
        rays_o, rays_d, near_plane = near, far_plane = far, 
        #alpha_fn = alpha_fn,
        render_step_size = step_size
    )
    if(ray_indices.shape[0] <= 0):
        return torch.zeros([num_pixels, 3]).cuda(), []
        #continue

    color, opacity, depth, extras = nerfacc.rendering(
        t_starts, t_ends, ray_indices, 
        n_rays = num_pixels, rgb_alpha_fn = rgb_alpha_fn
    )
    return color, ntk_records

