import argparse

from pykeen.pipeline import pipeline

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset split and model init.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../artifacts",
        help="Path to save model checkpoints and pipeline.",
    )

    parser.add_argument(
        '--test', 
        type=str, 
        help='path to test edges'
        )
    
    parser.add_argument(
        '--train', 
        type=str, 
        help='path to train edges'
        )
    
    parser.add_argument(
        '--valid', 
        type=str, 
        help='path to valid edges'
        )
    
    parser.add_argument(
        '--model', 
        type=str, 
        help='model name, rotate, transe, complex etc. Name must be a model in pykeen'
        )
    
    parser.add_argument(
        '--cuda', 
        type=str, 
        help='Select cuda device'
        )
    
    parser.add_argument(
        '--embedding_dim',
        default=304,
        type=int,
        help='Embedding size'
    )

    parser.add_argument(
        '--num_epoch',
        default=100,
        type=int,
        help='Num epoch to train'
    )

    parser.add_argument(
        '--lr',
        default=0.02,
        type=float,
        help='Learning rate'
    )

    return parser.parse_args()


def main() -> None:
    """Train a pykeen pipeline object using TransE on Hetionet."""

    args = get_args()

    result = pipeline(
        training=args.train,
        validation=args.valid, 
        testing=args.test,
        model=args.model,
        model_kwargs={"embedding_dim": args.embedding_dim, "random_seed": args.seed},
        training_kwargs={
            "num_epochs": args.num_epoch,
            "checkpoint_name": "transe_dev_checkpoint.pt",
            "checkpoint_directory": f"{args.save_path}/checkpoints/",
            "checkpoint_frequency": 0,
        },
        optimizer="Adagrad",
        loss="NSSA",
        optimizer_kwargs={"lr": args.lr},
        negative_sampler_kwargs={"num_negs_per_pos": 61},
        evaluator_kwargs={"filtered": True},
        random_seed=args.seed,
        device=args.cuda
    )

    result.save_to_directory(f"{args.save_path}/transe_hetnet")
    print("Pipeline Training Complete")


if __name__ == "__main__":

    main()
