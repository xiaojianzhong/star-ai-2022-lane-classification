import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv',
                        type=str,
                        help='csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)

    min = df['defect_prob'].min()
    max = df['defect_prob'].max()
    mean = df['defect_prob'].mean()
    std = df['defect_prob'].std()
    print(f'min: {min}')
    print(f'max: {max}')
    print(f'mean: {mean}')
    print(f'std: {std}')
    df['defect_prob'].hist()
    plt.show()


if __name__ == '__main__':
    main()
