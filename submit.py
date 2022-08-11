import argparse
import logging
import os
import warnings

import pandas as pd
import torch
from torch.nn import DataParallel
from tqdm import tqdm

from configs import CFG
from datas import build_dataset, build_dataloader
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

    # build dataset
    test_dataset = build_dataset('test')
    # build dataloader
    test_dataloader = build_dataloader(test_dataset, 'test')
    # build model
    model = build_model()
    model = DataParallel(model)
    model.cuda()

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError(f'checkpoint {args.checkpoint} not found')
    checkpoint = torch.load(args.checkpoint)
    model.module.load_state_dict(checkpoint['model']['state_dict'], strict=True)
    logging.info(f'load checkpoint {args.checkpoint}')

    df = pd.DataFrame(columns=['imagename', 'defect_prob'])

    with torch.no_grad():
        test_bar = tqdm(test_dataloader, desc='test', ascii=True)
        for sample in test_bar:
            x, name = sample['x'], sample['name'][0]
            x = x.cuda()

            y = model(x)
            y = y.squeeze(0)

            # prob = torch.max(torch.sigmoid(y), dim=0)[0].item()
            prob = torch.softmax(y, dim=0)[1].item()

            test_bar.set_postfix({
                'name': name,
                'probability': f'{prob:.2f}',
            })

            df.loc[len(df.index)] = [name, prob]
    df.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
