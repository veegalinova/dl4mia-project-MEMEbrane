from matplotlib.colors import ListedColormap
import numpy as np
import os
from pathlib import Path 

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data 
from torchvision 
from scipy.ndimage
from local import train
from unet import UNet
from tqdm import tqdm
import tifffile

from skimage.filters 
from data_processing import SDTDataset
    


def main():
    device = "cuda"  # 'cuda', 'cpu', 'mps'
    assert torch.cuda.is_available()

    transforms = transforms.Compose(
        [
            transforms.RandomCrop(256)
        ]
    )

    train_data = SDTDataset("nuclei_train_data", transform=transforms)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)

    learning_rate = 1e-4
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    # Initialize the model.
    unet = UNet(
        depth=4,
        in_channels=1,
        out_channels=1,
        final_activation="Tanh",
        num_fmaps=16,
        fmap_inc_factor=3,
        downsample_factor=2,
        padding="same",
        upsample_mode="nearest",
    )

    for epoch in range(20):
        train(
            unet,
            train_loader,
            optimizer,
            loss,
            epoch,
            log_interval=10,
            device=device,
        )

    output_path = Path("logs/")
    output_path.mkdir(exist_ok=True)
    torch.save(
        {'model_state_dict': unet.state_dict()},
        "logs/model.pth"
    )


if __name__ == "__main__":
    main()
