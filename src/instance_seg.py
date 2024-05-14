from matplotlib.colors import ListedColormap
import numpy as np
import os
from pathlib import Path 

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNet
from model_evaluation import validate
from tqdm import tqdm
import tifffile
from data_processing import SDTDataset
import sys
sys.path.append('.')


def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        if prediction.shape != y.shape:
            y = crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


def main():
    device = "cuda"  # 'cuda', 'cpu', 'mps'
    assert torch.cuda.is_available()

    transform = transforms.Compose(
        [
            transforms.CenterCrop((128, 128))
        ]
    )

    train_data = SDTDataset(transform=transform, train=True)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)
    val_data = SDTDataset(transform=transform, train=False, return_mask=True)
    val_loader = DataLoader(val_data, batch_size=5)

    print(len(train_loader), len(val_loader))
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

    learning_rate = 1e-4
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in tqdm(range(200)):
        train(
            unet,
            train_loader,
            optimizer,
            loss,
            epoch,
            log_interval=10,
            device=device,
        )
        metrics = validate(unet, val_loader, device=device, mode='sdt')
        scheduler.step(np.sum(metrics['mse_list']) / len(metrics['mse_list']))
        print(f"Validation mse after training epoch {epoch} is {np.sum(metrics['mse_list']) / len(metrics['mse_list'])}")

    output_path = Path("logs/")
    output_path.mkdir(exist_ok=True)
    torch.save(
        {'model_state_dict': unet.state_dict()},
        "logs/model.pth"
    )


if __name__ == "__main__":
    main()
