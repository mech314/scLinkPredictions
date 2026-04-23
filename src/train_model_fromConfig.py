import argparse
import json
from pykeen.pipeline import pipeline_from_path
from pathlib import Path
import tempfile

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--valid', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=314)
    return parser.parse_args()

def main():
    args = get_args()

    with open(args.config) as f:
        config = json.load(f)

    # patch paths
    config['pipeline']['training'] = args.train
    config['pipeline']['validation'] = args.valid
    config['pipeline']['testing'] = args.test

    # write patched config to temp file
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(config, tmp)
        tmp_path = tmp.name

    result = pipeline_from_path(tmp_path, device=args.device, random_seed=args.seed)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    result.save_to_directory(args.save_path)
    print(f"Saved to {args.save_path}")

if __name__ == '__main__':
    main()