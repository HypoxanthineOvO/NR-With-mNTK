import numpy as np
import math, torch, torchvision
import matplotlib.pyplot as plt
import warnings
import nerfacc

def load_datasets(dataset_size: int = 600, batch_size: int = 128):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), 
         torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    train_size, test_size = len(train_dataset), len(test_dataset)

    final_train_size = min(train_size, dataset_size)
    final_test_size = min(test_size, dataset_size // (train_size // test_size))
    train_dataset = torch.utils.data.Subset(train_dataset, range(final_train_size))
    test_dataset = torch.utils.data.Subset(test_dataset, range(final_test_size))
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 1
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 1
    )
    
    return train_dataset, test_dataset ,train_loader, test_loader

def dataset_to_tensor(dataset):
    return torch.stack([dataset[i][0] for i in range(len(dataset))])

def sample_in_datasets(
        dataset: torch.utils.data.Dataset, 
        num_samples: int = 5
    ) -> torch.Tensor:
    """
    从数据集中随机抽取指定数量的样本，并返回它们的张量表示。
    """
    indices = torch.randperm(len(dataset))[:num_samples]
    samples = [dataset[i][0] for i in indices]
    return torch.stack(samples)

def display_sample_images(
        dataloader: torch.utils.data.DataLoader, 
        num_images: int = 5, plt_height: int = 2
    ):
    # 获取一个批次的数据
    images, labels = next(iter(dataloader))
    
    # 确保 num_images 不超过该批次的实际图像数量
    num_images = min(num_images, len(images))

    plt.figure(figsize=(num_images * plt_height, plt_height))
    
    for i in range(num_images):
        oneimg = images[i]
        label = labels[i].item()
        
        if oneimg.shape[0] == 1:
            oneimg = oneimg.squeeze(0)
        else:
            oneimg = oneimg.permute(1, 2, 0)
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(oneimg, cmap='gray' if oneimg.ndim == 2 else None)
        plt.title(f'Label: {label}')
        plt.axis('off')
    
    plt.show()



def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0.5, 0.5, 0.5]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def Part_1_By_2(x: torch.tensor):
    x &= 0x000003ff;                 # x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x

def morton_naive(x: torch.tensor, y: torch.tensor, z: torch.tensor):
    return Part_1_By_2(x) + (Part_1_By_2(y) << 1) + (Part_1_By_2(z) << 2)

def morton(input):
    return morton_naive(input[..., 0], input[..., 1], input[..., 2])

def inv_Part_1_By_2(x: torch.tensor):
    x = ((x >> 2) | x) & 0x030C30C3
    x = ((x >> 4) | x) & 0x0300F00F
    x = ((x >> 8) | x) & 0x030000FF
    x = ((x >>16) | x) & 0x000003FF
    return x

def inv_morton_naive(input: torch.tensor):
    x = input &        0x09249249
    y = (input >> 1) & 0x09249249
    z = (input >> 2) & 0x09249249
    
    return inv_Part_1_By_2(x), inv_Part_1_By_2(y), inv_Part_1_By_2(z)

def inv_morton(input:torch.tensor):
    x,y,z = inv_morton_naive(input)
    return torch.stack([x,y,z], dim = -1)



def get_ray(x, y, hw, transform_matrix, focal, principal = [0.5, 0.5]):
    x = (x + 0.5) / hw[0]
    y = (y + 0.5) / hw[1]
    ray_o = transform_matrix[:3, 3]
    ray_d = np.array([
        (x - principal[0]) * hw[0] / focal,
        (y - principal[1]) * hw[1] / focal,
        1.0,
    ])
    ray_d = np.matmul(transform_matrix[:3, :3], ray_d)
    ray_d = ray_d / np.linalg.norm(ray_d)
    return ray_o, ray_d

class Camera:
    def __init__(self, resolution, camera_angle, camera_matrix):
        # Resolution: For Generate Image
        self.resolution = resolution
        self.w = self.resolution[0]
        self.h = self.resolution[1]
        self.image = np.zeros((resolution[0] * resolution[1], 3)) # RGB Image
        # Parameters
        self.position = np.array([0.0, 0.0, 0.0])
        self.camera_to_world = np.zeros((3, 3))
        self.focal_length = 1.0

        # Camera Coordinate Directions
        self.directions = None
        # Rays Origin and Direction
        self.rays_o = np.zeros((resolution[0], resolution[1], 3))
        self.rays_d = np.zeros((resolution[0], resolution[1], 3))

        assert camera_matrix.shape == (3, 4) or camera_matrix.shape == (4, 4)
        if(camera_matrix.shape == (4, 4)):
            camera_matrix = camera_matrix[:3]

        self.position = camera_matrix[:3, -1]
        self.camera_to_world = camera_matrix[:3, :3]
        self.w = self.resolution[0]
        self.h = self.resolution[1]
        self.focal_length = .5 * self.w / np.tan(.5 * camera_angle)
        # Generate Directions
        i, j = np.meshgrid(
            np.linspace(0, self.w-1, self.w), 
            np.linspace(0, self.h-1, self.h), 
            indexing='xy'
        )
        ngp_mat = nerf_matrix_to_ngp(camera_matrix)

        rays_o, rays_d = [], []
        for i in range(self.h):
            for j in range(self.w):
                ro, rd = get_ray(j, i, [self.h, self.w], ngp_mat, self.focal_length)
                rays_o.append(ro)
                rays_d.append(rd)
        
        self.rays_o = np.array(rays_o).reshape((-1, 3))
        self.rays_d = np.array(rays_d).reshape((-1, 3))
