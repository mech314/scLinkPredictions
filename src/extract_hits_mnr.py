import pandas as pd
import glob
import json
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings_dir', type=str, required=True, help='dir with ranking TSV files')
    parser.add_argument('--models_dir', type=str, required=True, help='dir with models/split_seedX/DATASET/results.json')
    parser.add_argument('--dataset', type=str, required=True, help='baseline or augmented')
    parser.add_argument('--output_dir', type=str, required=True, help='where to save CSVs')
    return parser.parse_args()

def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1-MNR
    mnr_data = []
    for f in glob.glob(f'{args.rankings_dir}/*.tsv'):
        name = f.split('/')[-1].replace('.tsv', '')
        parts = name.split('_')
        seed = parts[1]
        group = '_'.join(parts[2:])
        df = pd.read_csv(f, sep='\t')
        mnr = 1 - df['rank'].median()
        mnr_data.append({'seed': seed, 'group': group, 'mnr': mnr, 'n': len(df)})

    df_mnr = pd.DataFrame(mnr_data)
    summary = df_mnr.groupby('group')['mnr'].agg(['mean', 'std']).round(4)
    summary.columns = ['mnr_mean', 'mnr_std']
    summary = summary.sort_values('mnr_mean', ascending=False)
    print("=== 1-MNR by group ===")
    print(summary)
    summary.to_csv(f'{args.output_dir}/mnr_summary.csv')
    df_mnr.to_csv(f'{args.output_dir}/mnr_per_seed.csv', index=False)

    # Hits and MRR
    hits_data = []
    for f in glob.glob(f'{args.models_dir}/*/{ args.dataset}/results.json'):
        seed = f.split('/')[-3]
        with open(f) as fh:
            data = json.load(fh)
        b = data['metrics']['both']['realistic']
        hits_data.append({
            'seed': seed,
            'hits@1': b['hits_at_1'],
            'hits@10': b['hits_at_10'],
            'mrr': b['inverse_harmonic_mean_rank']
        })

    df_hits = pd.DataFrame(hits_data)
    print("\n=== Hits@1, Hits@10, MRR ===")
    print(df_hits.to_string(index=False))
    print(f"\nmean  hits@1={df_hits['hits@1'].mean():.4f}  hits@10={df_hits['hits@10'].mean():.4f}  mrr={df_hits['mrr'].mean():.4f}")
    print(f"std   hits@1={df_hits['hits@1'].std():.4f}  hits@10={df_hits['hits@10'].std():.4f}  mrr={df_hits['mrr'].std():.4f}")
    df_hits.to_csv(f'{args.output_dir}/hits_mrr.csv', index=False)

if __name__ == '__main__':
    main()