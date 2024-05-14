import numpy as np
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import tifffile


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
    """A PyTorch dataset to load images and cell masks."""

    def __init__(self, root_dir = "/group/dl4miacourse/projects/membrane/ecad_gfp_cropped/", 
    transform=None, img_transform=None, return_mask=False, train=False):
        
        # the directory with all the training samples
        if train:
            self.root_dir = root_dir + 'train/'
        else:
            self.root_dir = root_dir + 'test/'
        
        #the name of the raw images is different from the name of the mask names so we temporarilly store this info in separate variables
        self.list_images = os.listdir(self.root_dir+"/im/") 

        self.list_masks = os.listdir(self.root_dir+"/mask/") 

        assert len(self.list_images) == len(self.list_masks)

        #store information on the number of samples
        self.nsamples = len(self.list_images)

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

        self.loaded_imgs = [None] * self.nsamples
        self.loaded_masks = [None] * self.nsamples
        for sample_ind in range(self.nsamples):
            print(sample_ind)
            img_path = os.path.join(self.root_dir, "im", self.list_images[sample_ind])
            image = Image.open(img_path)
            image.load()
            self.loaded_imgs[sample_ind] = inp_transforms(image)

            embryo_info = self.list_images[sample_ind].split("_max_")[0]
            time_info = "_" + self.list_images[sample_ind].split("_")[-1]

            mask_filename = [i for i in self.list_masks if (embryo_info in i and time_info in i)][0]
            print(img_path)
            print(mask_filename)


            mask_path = os.path.join(self.root_dir, "mask", mask_filename)
            mask = Image.open(mask_path)
            mask.load()
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return self.nsamples

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

    def getImageList(self):
        return self.list_images

    def getMaskList(self):
        return self.list_masks


