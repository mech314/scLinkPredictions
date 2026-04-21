import sys
import os
os.environ['OMP_NUM_THREADS'] = '16' 
# this must be set before pykeen in imported so that it can take effect before things get started

from pykeen.pipeline import pipeline, pipeline_from_path
from pykeen.datasets.base import PathDataset
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from optuna.samplers import RandomSampler, QMCSampler
# import pipeline_from_path

import torch

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Run KGE hyperparameter optimization experiments')
    parser.add_argument('--id', type=str, help='id')
    parser.add_argument('--test', type=str, help='path to test edges')
    parser.add_argument('--train', type=str, help='path to train edges')
    parser.add_argument('--valid', type=str, help='path to valid edges')
    parser.add_argument('--model', type=str, help='model name, rotate, transe, complex etc. Name must be a model in pykeen')
    return parser.parse_args()


job_id = ''
if len(sys.argv) > 1:
    job_id = sys.argv[1]

args = get_args()
job_id = args.id

torch.manual_seed(42)

TEST_PATH =  args.test
TRAIN_PATH = args.train
VALID_PATH = args.valid

study_name = "{model}_{job_id}".format(model=args.model, job_id=job_id)  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name) # this for a sqlite database in the current working directory

hpo_pipeline_result = hpo_pipeline(
    n_trials=30,
    # n_trials=4,
    # sampler=RandomSampler, # this was the original one I used, commenting out to try something new
    sampler=QMCSampler,

    model_kwargs={
            "random_seed": 42,
        },

    model_kwargs_ranges=dict(
            embedding_dim=dict(
                type=int,
                low=64,
                high=512,
                q=16,
            ),
        ),

    training_kwargs_ranges=dict(
            num_epochs=dict(
                type=int,
                low=100,
                high=1000,
                q=100,
            ),
        ),
    
    optimizer_kwargs_ranges=dict(
            lr=dict(
                type=float,
                low=0.001,
                high=0.1,
                log=True,
            ),
        ),
    
    negative_sampler_kwargs_ranges=dict(
            num_negs_per_pos=dict(
                type=int,
                low=1,
                high=100,
                q=10,
            ),
        ),
    training=TRAIN_PATH,
    testing=TEST_PATH,
    validation=VALID_PATH,
    model=args.model,
    device='cuda',
    storage=storage_name,
    load_if_exists=True,
    study_name=study_name,
    stopper='early',
    stopper_kwargs={'frequency':10, 'patience':2, 'relative_delta':0.002},
    loss="NSSA",
)

hpo_pipeline_result.save_to_directory('PyKeenOut/{}'.format(study_name))

print('HPO pipeline result saved to PyKeenOut/{}'.format(study_name))
print('Training best model')
config = 'PyKeenOut/{}/best_pipeline/pipeline_config.json'.format(study_name)
# train best model
pipeline_result = pipeline_from_path(config,device='cuda')
pipeline_result.save_to_directory('PyKeenOut/{}/'.format(study_name))
