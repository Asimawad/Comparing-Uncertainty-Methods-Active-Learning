# Data utilities for loading datasets

import torch
from torchvision import datasets, transforms
import ddu_dirty_mnist
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64, eval_batch_size=1024, device="cuda"):
    train_dataset = ddu_dirty_mnist.DirtyMNIST('./data', train=True, download=True, device=device, normalize=True)
    test_dataset_mnist = ddu_dirty_mnist.FastMNIST('./data', train=False, download=True, device=device, normalize=True)
    test_dataset_dirty_mnist = ddu_dirty_mnist.DirtyMNIST('./data', train=False, download=True, device=device, normalize=True)
    test_dataset_fashion = datasets.FashionMNIST('./data', train=False, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,))
                                               ]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader_dirty_mnist = DataLoader(test_dataset_dirty_mnist, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    test_loader_fashion = DataLoader(test_dataset_fashion, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader_mnist, test_loader_dirty_mnist, test_loader_fashion