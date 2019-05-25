import torch.utils.data.dataset as Dataset

class Ultrasound_Dataset(Dataset):
    def __init__(self, images, root_dir):
        self.images = images
        self.root_dir = root_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
