from matplotlib.colors import ListedColormap
import numpy as np
import os
from pathlib import Path 

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import v2 as transformsv2
from model import UNet
from tqdm import tqdm
import tifffile
from data_processing import SDTDataset
import sys
sys.path.append('.')

def salt_and_pepper_noise(image, amount=0.05):
    """
    Add salt and pepper noise to an image
    """
    out = image.clone()
    num_salt = int(amount * image.numel() * 0.5)
    num_pepper = int(amount * image.numel() * 0.5)

    # Add Salt noise
    coords = [
        torch.randint(0, i - 1, [num_salt]) if i > 1 else [0] * num_salt
        for i in image.shape
    ]
    out[coords] = 1

    # Add Pepper noise
    coords = [
        torch.randint(0, i - 1, [num_pepper]) if i > 1 else [0] * num_pepper
        for i in image.shape
    ]
    out[coords] = 0

    return out

def gaussian_noise(image, mean = 0, var = 0.05):
    ch, row,col= image.shape
    mean = mean
    var = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(ch, row,col))
    gauss = gauss.reshape(ch,row,col)
    noisy = image + gauss
    noisy = transforms.Normalize([0.5], [0.5])(noisy)
    return noisy.to(torch.float)


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


def validate(model,
    loader,
    loss_function,
    epoch,
    tb_logger=None,
    device='cuda'):
    
    model.eval()
    running_loss = 0.

    i = 0
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(loader):
            # move input and target to the active device (either cpu or gpu)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss = loss_function(pred, y)
            running_loss += val_loss
            i += 1
    val_loss /= i
    if tb_logger is not None:
        tb_logger.add_scalar("MSE/validation", val_loss, epoch)
    print(f"Validation mse after training epoch {epoch} is {val_loss}")
    
    return val_loss



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    device = "cuda"  # 'cuda', 'cpu', 'mps'
    assert torch.cuda.is_available()

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),  
            transforms.RandomRotation([90,90]),
            transforms.RandomCrop(256)   
            ]
    )
    img_transforms = transforms.Compose(
        [
            transforms.GaussianBlur(kernel_size = 5, sigma= 5),
            transformsv2.Lambda(salt_and_pepper_noise),
            transformsv2.Lambda(gaussian_noise)
        ]
    )

    train_data = SDTDataset(transform=transform, img_transform=img_transforms, train=True)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=8)
    val_data = SDTDataset(transform=transform, img_transform=img_transforms, train=False, return_mask=True)
    val_loader = DataLoader(val_data, batch_size=5)

    print(len(train_loader), len(val_loader))
    # Initialize the model.
    unet = UNet(
        depth=4,
        in_channels=1,
        out_channels=1,
        final_activation="Tanh",
        num_fmaps=16,
        fmap_inc_factor=2,
        downsample_factor=2,
        padding="same",
        upsample_mode="nearest",
    )

    learning_rate = 1e-4
    loss = torch.nn.MSELoss()
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-8)

    early_stopper = EarlyStopper(patience=10)
    for epoch in tqdm(range(200)):
        train(
            unet,
            train_loader,
            optimizer,
            loss,
            epoch,
            log_interval=10,
            device=device,
            tb_logger=writer,
        )
        val_loss = validate(unet, 
                            train_loader,
                            loss,
                            epoch, 
                            writer,
                            device='cuda')

        scheduler.step(val_loss)
        if early_stopper.early_stop(val_loss):
            print("Stopping test early!")
            break

    output_path = Path("logs/")
    output_path.mkdir(exist_ok=True)
    torch.save(
        {'model_state_dict': unet.state_dict()},
        "logs/model.pth"
    )

    writer.flush()


if __name__ == "__main__":
    main()
