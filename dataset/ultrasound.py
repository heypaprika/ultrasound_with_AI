from torch.utils.data.dataset import Dataset as Dataset

# image와 label을 동시에 가지고 있는 것이어야 한다.

class Ultrasound_Dataset(Dataset):
    def __init__(self, images, root_dir):
        self.images = images
        self.root_dir = root_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]


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

