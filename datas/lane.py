import os

from PIL import Image
from torch.utils.data import Dataset


class StarAI2022LaneDataset(Dataset):
    def __init__(self, root, transform=None):
        super(StarAI2022LaneDataset, self).__init__()

        self.image_paths = []
        self.labels = []
        self.gts = []

        images_dir = os.path.join(root, 'train_image', 'labeled_data')
        csv_path = os.path.join(root, 'train_label', 'train_label.csv')
        with open(csv_path) as f:
            for line in f.read().splitlines():
                values = line.split(',')
                self.image_paths.append(os.path.join(images_dir, values[0]))
                self.labels.append(int(values[1]))

                if values[1:][0] == '0':
                    gt = 0
                else:
                    gt = 1
                # gt = np.zeros(7, dtype=np.int)
                # for value in values[1:]:
                #     value = int(value)
                #     if value != 0:
                #         gt[value - 1] = 1
                self.gts.append(gt)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        gt = self.gts[i]

        return image, gt

    def get_labels(self):
        return self.labels
