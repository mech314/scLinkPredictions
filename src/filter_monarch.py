"""
Will filter Monarch Graph to contain only desired nodes.
"""

import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser('Arguments')

    parser.add_argument('--input', required=True, help='Path to Monarch KG')
    parser.add_argument('--nodes', default="MONDO:,HGNC:,HP:", help='Comma separated prefixes to retain')
    parser.add_argument('--output', required=True, help='Path where to save filtered file')

    return parser.parse_args()


def main():

    args = get_args()

    df = pd.read_csv(args.input, sep="\t", usecols=["subject", "predicate", "object"])

    nodes = tuple(args.nodes.split(","))
    mask = (
        df["subject"].str.startswith(nodes) &
        df["object"].str.startswith(nodes)
    )

    df_filtered = df[mask]
    print(f"Before: {len(df):,}  After: {len(df_filtered):,}")

    df_filtered = df_filtered[["subject", "predicate", "object"]]
    df_filtered.to_csv(args.output, sep="\t", index=False, header=False)

if __name__ == '__main__':
    main()