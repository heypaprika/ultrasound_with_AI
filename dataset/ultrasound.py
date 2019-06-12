from torch.utils.data.dataset import Dataset as Dataset
import os
import cv2
import torch
import numpy as np
# image와 label을 동시에 가지고 있는 것이어야 한다.

class Ultrasound_Dataset(Dataset):
    def __init__(self, images, label_path, image_path, transforms=None):
        self.images = images
        y_data = np.loadtxt(label_path, delimiter=' ', dtype=np.float32)
        y_data[:, 0] /= 10000000
        y_data[:, 1] /= 0.0015
        y_data[:, 2] /= 0.00075
        y_data[:, 3] /= 0.01
        y_data[:, 4] /= 0.000025

        self.y_data = torch.from_numpy(y_data[:, :])
        self.image_path = image_path
        self.transforms = transforms
        self.array = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        im = cv2.imread(os.path.join(self.image_path, image),1)
        if self.transforms is not None:
            im = self.transforms(im)
        label = self.y_data[index]
        return (im, label)

class Ultrasound_Dataset_test(Dataset):
    def __init__(self, images, image_path, transforms=None):
        self.images = images
        self.image_path = image_path
        self.transforms = transforms
        self.array = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        im = cv2.imread(os.path.join(self.image_path, image),1)
        if self.transforms is not None:
            im = self.transforms(im)
        return (im)


class Ultrasound_Dataset_for_dump(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return (image,label)

