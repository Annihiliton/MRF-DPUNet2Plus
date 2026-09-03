import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def quarter_turn_flip(image, mask):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    return np.flip(image, axis=axis).copy(), np.flip(mask, axis=axis).copy()


def small_angle_rotate(image, mask):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return image, mask


def image_tensor(image):
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float().div(255.0)


def mask_tensor(mask):
    return torch.from_numpy(np.ascontiguousarray(mask)).long()


class TrainTransform:
    def __call__(self, sample):
        image = sample["image"]
        mask = sample["label"]
        if random.random() > 0.5:
            image, mask = quarter_turn_flip(image, mask)
        elif random.random() < 0.5:
            image, mask = small_angle_rotate(image, mask)
        return {"image": image_tensor(image), "label": mask_tensor(mask)}


class EvalTransform:
    def __call__(self, sample):
        return {"image": image_tensor(sample["image"]), "label": mask_tensor(sample["label"])}


class SegmentationDataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_size=224):
        self.image_size = image_size
        self.image_path = Path(dataset_path) / "img"
        self.mask_path = Path(dataset_path) / "label"
        self.transform = transform
        if not self.image_path.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_path}")
        if not self.mask_path.is_dir():
            raise FileNotFoundError(f"Label folder not found: {self.mask_path}")
        self.image_list = sorted(p.name for p in self.image_path.iterdir() if p.is_file())
        if not self.image_list:
            raise RuntimeError(f"No images found in: {self.image_path}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = self.image_path / image_name
        mask_file = self.mask_path / f"{Path(image_name).stem}.png"
        image = cv2.imread(os.fspath(image_file), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.fspath(mask_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_file}")
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_file}")
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 0).astype(np.uint8)
        sample = {"image": image, "label": mask}
        sample = self.transform(sample) if self.transform is not None else EvalTransform()(sample)
        return sample, image_name
