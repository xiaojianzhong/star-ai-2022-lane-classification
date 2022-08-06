import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        type=str,
                        help='directory')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()

    size2num = dict()
    means = np.zeros(3)
    stds = np.zeros(3)

    num_images = len(os.listdir(args.dir))
    print(f'number of images: {num_images}')

    bar = tqdm(sorted(os.listdir(args.dir)), ascii=True)
    for image_name in bar:
        image_path = os.path.join(args.dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if image.size not in size2num:
            size2num[image.size] = 0
        size2num[image.size] += 1
        x = np.array(image)
        mean = x.mean(axis=(0, 1))
        means += mean
        std = x.std(axis=(0, 1))
        stds += std

        bar.set_postfix({
            'name': image_name,
            'size': image.size,
            'mean': mean,
            'std': std,
        })
    print(size2num)
    means /= num_images
    stds /= num_images
    print(means)
    print(stds)


if __name__ == '__main__':
    main()
