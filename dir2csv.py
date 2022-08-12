import argparse
import os

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        type=str,
                        help='directory')
    parser.add_argument('csv',
                        type=str,
                        help='csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    df = pd.DataFrame(columns=[0, 1])
    bar = tqdm(sorted(os.listdir(args.dir)), ascii=True)
    for image_name in bar:
        df.loc[len(df.index)] = [image_name, 0]
    df.to_csv(args.csv, header=False, index=False)


if __name__ == '__main__':
    main()
