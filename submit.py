import argparse
import logging
import os
import warnings

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from configs import CFG
from datas import build_transform
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--csv',
                        type=str,
                        default='submission.csv',
                        help='csv name')
    parser.add_argument('--configs',
                        nargs='+',
                        default=[],
                        help='custom configs')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()

    # log to stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
        ])

    # ignore warnings
    warnings.filterwarnings('ignore')

    # merge config with config file
    CFG.merge_from_file(args.config)
    # merge config with custom configs from command line arguments
    CFG.merge_from_list(args.configs)

    # build transform
    transform = build_transform()

    # build model
    model = build_model()
    model.cuda()

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError(f'checkpoint {args.checkpoint} not found')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model']['state_dict'], strict=True)
    logging.info(f'load checkpoint {args.checkpoint}')

    df = pd.DataFrame(columns=['imagename', 'defect_prob'])

    images_dir = os.path.join(CFG.DATASET.ROOT, 'test_images')
    bar = tqdm(sorted(os.listdir(images_dir)), ascii=True)
    for image_name in bar:
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        x = image
        x = x.cuda()
        x = x.unsqueeze(0)

        y = model(x)
        y = y.squeeze(0)
        # prob = torch.max(torch.sigmoid(y), dim=0)[0].item()
        prob = torch.softmax(y, dim=0)[1].item()

        bar.set_postfix({
            'name': image_name,
            'probability': f'{prob:.2f}',
        })

        df.loc[len(df.index)] = [image_name, prob]
    df.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
