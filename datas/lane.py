import os

from PIL import Image
from torch.utils.data import Dataset


class StarAI2022LaneDataset(Dataset):
    def __init__(self, root, split, transform=None):
        super(StarAI2022LaneDataset, self).__init__()

        self.image_names = []
        self.image_paths = []
        self.labels = []
        self.gts = []

        if split == 'train':
            images_dir = os.path.join(root, 'train_image', 'labeled_data')
            csv_path = os.path.join(root, 'train_label', 'labeled_data', 'train_label.csv')
        elif split == 'test':
            images_dir = os.path.join(root, 'test_images')
            csv_path = os.path.join(root, 'test_label', 'test_label.csv')
        else:
            raise NotImplementedError(f'invalid split: {split}')

        with open(csv_path) as f:
            for line in f.read().splitlines():
                values = line.split(',')
                self.image_names.append(values[0])
                self.image_paths.append(os.path.join(images_dir, values[0]))

                if values[1:][0] == '0':
                    self.labels.append(0)
                    self.gts.append(0)
                else:
                    self.labels.append(1)
                    self.gts.append(1)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_name = self.image_names[i]
        image_path = self.image_paths[i]
        image = Image.open(image_path).convert('RGB')
        x = image
        if self.transform is not None:
            x = self.transform(x)

        gt = self.gts[i]

        return {
            'name': image_name,
            'path': image_path,
            'x': x,
            'gt': gt,
        }

    def get_labels(self):
        return self.labels
