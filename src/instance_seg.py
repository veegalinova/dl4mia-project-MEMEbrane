from matplotlib.colors import ListedColormap
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from local import train, NucleiDataset, plot_two, plot_three, plot_four
from unet import UNet
from tqdm import tqdm
import tifffile

from skimage.filters import threshold_otsu


#SIGNED DISTANCE SEGMENTATION

def compute_sdt(labels: np.ndarray, scale: int = 5):
    """Function to compute a signed distance transform."""

    # compute the distance transform inside and outside of the objects
    labels = np.asarray(labels)
    ids = np.unique(labels) 
    ids = ids[ids != 0]
    inner = np.zeros(labels.shape, dtype=np.float32)

    for id_ in ids:
        inner += distance_transform_edt(labels == id_)
    outer = distance_transform_edt(labels == 0)

    # create the signed distance transform
    distance = inner - outer

    # scale the distances so that they are between -1 and 1 (hint: np.tanh)
    distance = np.tanh(distance / scale) 

    # be sure to return your solution as type 'float'
    return distance.astype(float)




class SDTDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks."""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = (
            "/group/dl4miacourse/segmentation/" + root_dir
        )  # the directory with all the training samples
        self.samples = os.listdir(self.root_dir)  # list the samples
        self.return_mask = return_mask
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )

        self.loaded_imgs = [None] * len(self.samples)
        self.loaded_masks = [None] * len(self.samples)
        for sample_ind in range(len(self.samples)):
            img_path = os.path.join(
                self.root_dir, self.samples[sample_ind], "image.tif"
            )
            image = Image.open(img_path)
            image.load()
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(
                self.root_dir, self.samples[sample_ind], "label.tif"
            )
            mask = Image.open(mask_path)
            mask.load()
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        sdt = self.create_sdt_target(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.return_mask is True:
            return image, transforms.ToTensor()(mask), sdt
        else:
            return image, sdt

    def create_sdt_target(self, mask):
        sdt_target_array = compute_sdt(mask)
        sdt_target = transforms.ToTensor()(sdt_target_array)
        return sdt_target.float()
    


def main():
    device = "cuda"  # 'cuda', 'cpu', 'mps'
    assert torch.cuda.is_available()
    label_cmap = ListedColormap(np.load("/group/dl4miacourse/segmentation/cmap_60.npy"))

    root_dir = "/group/dl4miacourse/segmentation/nuclei_train_data"  # the directory with all the training samples
    samples = os.listdir(root_dir)
    idx = np.random.randint(len(samples))  # take a random sample.
    img = tifffile.imread(
        os.path.join(root_dir, samples[idx], "image.tif")
    )  # get the image
    label = tifffile.imread(
        os.path.join(root_dir, samples[idx], "label.tif")
    )  # get the image
    sdt = compute_sdt(label)
    plot_two(img, sdt, label="SDT")

    train_data = SDTDataset("nuclei_train_data", transforms.RandomCrop(256))
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)

    idx = np.random.randint(len(train_data))  # take a random sample
    img, sdt = train_data[idx]  # get the image and the nuclei masks
    plot_two(img[0], sdt[0], label="SDT")


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

    val_data = SDTDataset("nuclei_val_data")
    unet.eval()
    idx = np.random.randint(len(val_data))  # take a random sample.
    image, sdt = val_data[idx]  # get the image and the nuclei masks.
    image = image.to(device)
    pred = unet(torch.unsqueeze(image, dim=0))
    image = np.squeeze(image.cpu())
    sdt = np.squeeze(sdt.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())
    plot_three(image, sdt, pred)
