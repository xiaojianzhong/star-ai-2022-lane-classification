import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs',
                        nargs='+',
                        default=[],
                        help='input csvs')
    parser.add_argument('--merge-by',
                        type=str,
                        default='vote',
                        choices=[
                            'vote',
                            'arithmetic-mean',
                            'geometric-mean',
                            'harmonic-mean',
                            'quadratic-mean',
                        ],
                        help='method for merging csvs')
    parser.add_argument('--out-csv',
                        type=str,
                        default='submission.csv',
                        help='output csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    n = len(args.csvs)
    dfs = [pd.read_csv(csv) for csv in args.csvs]

    df = pd.DataFrame(index=dfs[0].index, columns=dfs[0].columns)
    for i, _ in df.iterrows():
        df.loc[i, 'imagename'] = dfs[0].loc[i, 'imagename']

        probs = np.empty(n)
        for j, df in enumerate(dfs):
            prob = df.loc[i, 'defect_prob']
            probs[j] = prob

        if args.merge_by == 'vote':  # 投票
            num_positive = (probs > 0.5).sum()
            num_negative = (probs < 0.5).sum()
            if num_positive > num_negative:
                df.loc[i, 'defect_prob'] = probs.max()
            else:
                df.loc[i, 'defect_prob'] = probs.min()
        elif args.merge_by == 'arithmetic-mean':  # 算数平均
            df.loc[i, 'defect_prob'] = probs.mean()
        elif args.merge_by == 'geometric-mean':  # 几何平均
            df.loc[i, 'defect_prob'] = probs.prod() ** (1.0 / n)
        elif args.merge_by == 'harmonic-mean':  # 调和平均
            df.loc[i, 'defect_prob'] = n / (1 / probs).sum()
        elif args.merge_by == 'quadratic-mean':  # 平方平均
            df.loc[i, 'defect_prob'] = np.sqrt((probs ** 2).sum() / n)
        else:
            raise NotImplementedError(f'invalid method for merging csvs: {args.merge_by}')
    df.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()
