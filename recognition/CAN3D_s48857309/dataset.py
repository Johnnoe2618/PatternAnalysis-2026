"""This module allows loading of the HipMRI database.

References:
    load_data_3D: taken from the assignment sheet
"""
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit



def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1

    return res


# load medical image functions
def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    early_stop=False,
):
    """
    Load medical image data from names, cases list provided into a list for
    each .

    This function pre - allocates 5D arrays for conv3d to avoid excessive
    memory usage .

    normImage : bool ( normalise the image 0.0-1.0)
    dtype : Type of the data . If dtype =np.uint8 , it is assumed that the data
    is labels
    early_stop : Stop loading pre - maturely ? Leaves arrays mostly empty , for
    quick loading and testing scripts .
    """

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching="unchanged")
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2], :] = (
                inImage
            )
        else:
            images[i, : inImage.shape[0], : inImage.shape[1], : inImage.shape[2]] = (
                inImage
            )

        if i > 20 and early_stop:
            break

    return images

class HipMRI3D(Dataset):
    def __init__(self, images_path='/home/groups/comp3710/HipMRI_Study_open/semantic_MRs', labels_path='/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only', early_stop=True):
        self.image_dir = Path(images_path)
        self.labels_dir = Path(labels_path)
        self.image_paths = list(self.image_dir.glob('*.nii.gz'))
        self.labels_paths = [(self.labels_dir / "{}_SEMANTIC".format(image_path.name[0:10])).with_suffix(".nii.gz") for image_path in self.image_paths]
        self.patient_labels = [image.name[0:4] for image in self.image_paths]

        self.images = load_data_3D(self.image_paths, True, False, np.float32, False, early_stop)
        self.labels = load_data_3D(self.labels_paths, False, True, np.uint8, False, early_stop)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __getitems__(self, idxs):
        return self.images[idxs], self.labels[idxs]

def get_training_test_split(dataset: HipMRI3D):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(dataset.images, groups=dataset.patient_labels))

    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, test_set
