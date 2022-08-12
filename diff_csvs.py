import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs',
                        nargs='+',
                        default=[],
                        help='csvs')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    n = len(args.csvs)
    dfs = [pd.read_csv(csv) for csv in args.csvs]
    num_rows = len(dfs[0])
    for df in dfs:
        assert num_rows == len(df)

    for i in range(n):
        for j in range(i+1, n):
            df1, df2 = dfs[i], dfs[j]

            print(f'{args.csvs[i]} <-> {args.csvs[j]}')

            infos = []
            for k in range(num_rows):
                image_name1, prob1 = df1.loc[k, 'imagename'], df1.loc[k, 'defect_prob']
                image_name2, prob2 = df2.loc[k, 'imagename'], df2.loc[k, 'defect_prob']
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

            min_distance = min(distances)
            max_distance = max(distances)
            avg_distance = sum(distances) / len(infos)
            print(f'\tminimum distance: {min_distance}')
            print(f'\tmaximum distance: {max_distance}')
            print(f'\taverage distance: {avg_distance}')


if __name__ == '__main__':
    main()
