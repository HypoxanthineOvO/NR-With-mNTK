import os
import json
import cv2 as cv
import numpy as np
from tqdm import trange
import torch
import nerfacc
import msgpack
from utils import nerf_matrix_to_ngp, inv_morton_naive

class NeRFSynthetic:
    def __init__(self, data_dir):
        self.scale = 0.33
        self.offset = [0.5, 0.5, 0.5]
        images = []
        transform_matrixs = []
        split = {}
        for data_type in ["train", "test", "val"]:
            print(f"Loading {data_type.capitalize()} Data...")
            with open(os.path.join(data_dir, f"transforms_{data_type}.json"), "r") as f:
                meta = json.load(f)
            
            split[data_type] = len(meta["frames"])
            for i in trange(len(meta["frames"])):
                frame = meta["frames"][i]
                file_name = os.path.join(data_dir, frame["file_path"] + ".png")
                image_raw = cv.imread(file_name, cv.IMREAD_UNCHANGED)
                image_raw = cv.cvtColor(image_raw, cv.COLOR_BGRA2RGBA)
                image = image_raw / 255.
                image = image[..., :3] * image[..., 3:]
                transform_matrix = nerf_matrix_to_ngp(
                    torch.tensor(frame["transform_matrix"]), self.scale, self.offset
                )
                images.append(image)
                transform_matrixs.append(transform_matrix)
        self.split = split
        images_np = np.array(images)
        self.images = torch.tensor(images_np, dtype = torch.float32)
        
        self.transform_matrixs = torch.tensor(np.array(transform_matrixs))[:, :3, :]
        
        H, W = self.images.shape[1:3]
        camera_angle = float(meta["camera_angle_x"])
        self.focal = torch.tensor([
            0.5 * W / np.tan(0.5 * camera_angle),
            0.5 * H / np.tan(0.5 * camera_angle)
        ])
        self.principal = torch.tensor([0.5, 0.5])
        self.HW = torch.tensor([H, W])
    def test(self, test_id = 0):
        id = self.split["train"] + test_id
        
        ref_images = self.images[id]
        c2w = self.transform_matrixs[id]
        w_ids, h_ids = np.meshgrid(
            np.linspace(0, self.HW[1]-1, self.HW[1]), 
            np.linspace(0, self.HW[0]-1, self.HW[0]), 
            indexing='xy'
        )
        w_ids = w_ids.reshape([-1,])
        h_ids = h_ids.reshape([-1,])
        hw_ids = np.random.randint(0, 800 * 800, (800 * 800, 1))
        w_id = torch.tensor(w_ids[hw_ids], dtype = torch.float32)
        h_id = torch.tensor(h_ids[hw_ids], dtype = torch.float32)
        ray_o, ray_d = torch.vmap(
            self.generate_ray,
            in_dims = (None, 0, 0, None, None, None)
        )(c2w, w_id, h_id, self.focal, self.principal, self.HW)
        return ref_images, ray_o, ray_d
    
    def sample(self, batch_size):
        image_id = torch.tensor(np.random.randint(0, self.split["train"] + self.split["test"] + self.split["val"]))
        transform_matrixs = self.transform_matrixs[image_id]#.expand([batch_size] + list(self.transform_matrixs[0].shape))
        image = self.images[image_id].reshape([-1, 3])
        
        w_ids, h_ids = np.meshgrid(
            np.linspace(0, self.HW[1]-1, self.HW[1]), 
            np.linspace(0, self.HW[0]-1, self.HW[0]), 
            indexing='xy'
        )
        w_ids = w_ids.reshape([-1,])
        h_ids = h_ids.reshape([-1,])
        hw_ids = np.random.randint(0, 800 * 800, (batch_size, 1))
        w_id = torch.tensor(w_ids[hw_ids], dtype = torch.float32)
        h_id = torch.tensor(h_ids[hw_ids], dtype = torch.float32)
        
        pixels = image[hw_ids].reshape([batch_size, 3])
        
        ray_o, ray_d = torch.vmap(
            self.generate_ray,
            in_dims = (None, 0, 0, None, None, None)
        )(transform_matrixs, w_id, h_id, self.focal, self.principal, self.HW)
        
        return pixels, ray_o, ray_d
    


    @staticmethod
    def generate_ray(transform_matrix, x, y, focal, principal, hw):
        x = (x + 0.5) / hw[0]
        y = (y + 0.5) / hw[1]
        ray_o = transform_matrix[:, 3]
        ray_d_x: torch.Tensor = (x - principal[0]) * hw[0] / focal[0]
        ray_d_y: torch.Tensor = (y - principal[1]) * hw[1] / focal[1]
        ray_d_z: torch.Tensor = torch.ones_like(ray_d_x)
        ray_d = torch.cat([ray_d_x, ray_d_y, ray_d_z], dim = -1)
        ray_d = torch.matmul(transform_matrix[:3, :3], ray_d)
        ray_d = ray_d / torch.norm(ray_d)
        return (ray_o, ray_d)


def load_msgpack(path: str):
    print(f"Loding Msgpack from {path}")
    # Return Value
    res = {}
    # Get Morton3D Object
    
    # Set File Path
    assert (os.path.isfile(path))
    dir_name, full_file_name = os.path.split(path)
    file_name, ext_name = os.path.splitext(full_file_name)
    # Load the msgpack
    with open(path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
        config = next(unpacker)

    # Set Model Parameters
    # Total: 12206480 Parameters
    params_binary = np.frombuffer(config["snapshot"]["params_binary"], dtype = np.float16, offset = 0)
    # Transform to torch tensor
    params_binary = torch.tensor(params_binary, dtype = torch.float32)
    # Generate Parameters Dictionary
    params = {}
    # Params for Hash Encoding Network
    ## Network Params Size: 32 * 64 + 64 * 16 = 3072
    hashenc_params_network = params_binary[:(32 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 16):]
    # Params for RGB Network
    ## Network Params Size: 32 * 64 + 64 * 64 + 64 * 16 = 7168
    rgb_params_network = params_binary[:(32 * 64 + 64 * 64 + 64 * 16)]
    params_binary = params_binary[(32 * 64 + 64 * 64 + 64 * 16):]
    # Params for Hash Encoding Grid
    ## Grid size: 12196240
    hashenc_params_grid = params_binary

    # Generate Final Parameters
    params["HashEncoding"] = torch.concat([hashenc_params_network, hashenc_params_grid,  ])
    params["RGB"] = rgb_params_network
    res["params"] = params
    # Occupancy Grid Part
    grid_raw = torch.tensor(
        np.frombuffer(config["snapshot"]["density_grid_binary"],dtype=np.float16).astype(np.float32),
        dtype = torch.float32
        )

    grid = torch.zeros([128 * 128 * 128], dtype = torch.float32)

    x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
    grid[x * 128 * 128 + y * 128 + z] = grid_raw
    grid_3d = torch.reshape(grid > 0.01, [1, 128, 128, 128]).type(torch.bool)

    params_grid = {
        "resolution": torch.tensor([128, 128, 128], dtype = torch.int32),
        "aabbs": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        "occs":grid,
        "binaries": grid_3d
    }
    
    res["OccupancyGrid"] = params_grid
    
    print("Msgpack Loaded!")
    return res


def load_density_grid(path: str, params_config_path: str = None):
    print(f"Loding Msgpack from {path}")
    # Return Value
    res = {}
    # Get Morton3D Object
    
    # Set File Path
    assert (os.path.isfile(path)), f"File {path} not found."
    dir_name, full_file_name = os.path.split(path)
    file_name, ext_name = os.path.splitext(full_file_name)
    # Load the msgpack
    with open(path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
        config = next(unpacker)

    snapshot = config["snapshot"]
    density_grid_size = snapshot["density_grid_size"]
    aabb_min = snapshot["aabb"]["min"]
    aabb_max = snapshot["aabb"]["max"]
    #exit()
    grid_raw = torch.tensor(
        np.frombuffer(snapshot["density_grid_binary"],dtype=np.float16).astype(np.float32),
        dtype = torch.float32, device = "cuda"
        )

    num_of_layers = grid_raw.shape[0] // (density_grid_size ** 3)
    #grid_raw = torch.ones_like(grid_raw, dtype = torch.float32, device = 'cuda')
    total_grid = []
    total_grid_binary = []
    aabbs = []
    for i in range(num_of_layers):
        aabb_left = 0.5 - 0.5 * (2 ** i)
        aabb_right = 0.5 + 0.5 * (2 ** i)
        aabbs.append([aabb_left, aabb_left, aabb_left, aabb_right, aabb_right, aabb_right])
        grid = torch.zeros([density_grid_size ** 3], dtype = torch.float32, device = 'cuda')

        x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
        grid[x * 128 * 128 + y * 128 + z] = grid_raw[i * (density_grid_size ** 3): (i + 1) * (density_grid_size ** 3)]
        grid_3d = torch.reshape(grid > 0.01, [1, density_grid_size, density_grid_size, density_grid_size]).type(torch.bool)
        total_grid.append(grid)
        total_grid_binary.append(grid_3d)

    total_grid = torch.cat(total_grid, dim = 0)
    total_grid_binary = torch.cat(total_grid_binary, dim = 0)
    estimator = nerfacc.OccGridEstimator(
        aabb_min + aabb_max,
        resolution = 128, levels = num_of_layers
    )
    params_grid = {
        "resolution": torch.tensor([density_grid_size, density_grid_size, density_grid_size], dtype = torch.int32),
        #"aabbs": torch.tensor([[-0.5, -0.5, -0.5, 1.5, 1.5, 1.5]]),
        "aabbs": torch.tensor(aabbs),
        "occs":total_grid,
        "binaries": total_grid_binary#grid_3d
    }
    estimator.load_state_dict(params_grid)
    
    return estimator