import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv1',
                        type=str,
                        help='first csv')
    parser.add_argument('csv2',
                        type=str,
                        help='second csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    assert len(df1) == len(df2)
    num_rows = len(df1)

    infos = []
    for i in range(num_rows):
        image_name1, prob1 = df1.loc[i, 'imagename'], df1.loc[i, 'defect_prob']
        image_name2, prob2 = df2.loc[i, 'imagename'], df2.loc[i, 'defect_prob']
        assert image_name1 == image_name2
        image_name = image_name1
        distance = abs(prob1 - prob2)
        infos.append({
            'name': image_name,
            'distance': distance,
            'probability1': prob1,
            'probability2': prob2,
        })
    info2distance = lambda info: info['distance']
    distances = list(map(info2distance, infos))
    plt.hist(np.array(distances), bins=4)
    plt.show()

    print(f'{args.csv1} <-> {args.csv2}')
    min_distance = min(distances)
    max_distance = max(distances)
    avg_distance = sum(distances) / len(infos)
    print(f'\tminimum distance: {min_distance}')
    print(f'\tmaximum distance: {max_distance}')
    print(f'\taverage distance: {avg_distance}')


if __name__ == '__main__':
    main()
