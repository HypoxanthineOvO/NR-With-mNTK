import torch, torchvision
import matplotlib.pyplot as plt
import warnings

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