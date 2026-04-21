"""
Params:
a_terms - list of terms in group A, terms must be in the training set
b_terms - list of terms in group B, terms must be in the training set
a_label - label for group A
b_label - label for group B
relation - relation to use for predictions
prediction_target - "head" or "tail", which entity are you trying to predict? Edges are assumed to be directed.
prediction_prefix - the type of entity to predict, the prefix of the entity id eg "MONDO:" "HGNC:"
train_triples - training triple list 
validation_triples - validation triple list
test_triples - test triple list
model - trained pykeen model in .pkl format
output_prefix - prefix for output files
"""

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from pykeen.datasets.base import PathDataset
from scipy.stats import kruskal
from typing import List
from pykeen import predict
import os
import pickle
import warnings

def get_args():
    parser = argparse.ArgumentParser(description='Run the test-omatic experiment')
    parser.add_argument('--a_terms', type=str, required=True, help='list of terms in group A, terms must be in the training set')
    parser.add_argument('--a_label', type=str, required=True, help='label for group A')
    parser.add_argument('--relation', type=str, required=True, help='relation to use for predictions')
    parser.add_argument('--prediction_target', type=str, required=True, help='"head" or "tail", which entity are you trying to predict? Edges are assumed to be directed.')
    parser.add_argument('--prediction_prefix', type=str, required=True, help='the type of entity to predict, the prefix of the entity id eg "MONDO:" "HGNC:"')
    parser.add_argument('--train_triples', type=str, required=True, help='training triple list')
    parser.add_argument('--validation_triples', type=str, required=True, help='validation triple list')
    parser.add_argument('--test_triples', type=str, required=True, help='test triple list')
    parser.add_argument('--model', type=str, required=True, help='trained pykeen model in .pkl format')
    parser.add_argument('--output', type=str, required=True, help='output file, tsv')
    parser.add_argument('--progress_bar', dest='progress_bar', action='store_true',default=False, help='show progress bar')
    # param for threads, default is 1
    parser.add_argument('--threads', type=int, default=1, help='number of threads to use')
    args = parser.parse_args()
    return args

def get_test_edges_for_term(terms: List[str],
                            relation: str,
                            test_triples: List[List[str]],
                            progress_bar=False) -> List[List[str]]:
    """
    take a list of terms, relation type and return all edges in the test set that contain the term and relation
    """
    # pre filter test_triples to only have edges that contain the relation, this is a HUGE speed up
    filtered_test_triples = [edge for edge in test_triples if relation in edge]
    terms_test_edges = []
    for i,term in enumerate(terms):
        # print out a progress bar update every 5% complete
        if i % (len(terms) // 20) == 0 and progress_bar:
            print(f'{i} / {len(terms)}')
        # find all edges that contain the term and relation
        terms_test_edges += [edge for edge in filtered_test_triples if term in edge]
            
    return terms_test_edges

def get_scores_for_edges(terms: List[str],
                         relation: str,
                         model,
                         degs: dict,
                         train_triples: List[List[str]],
                         test_triples: List[List[str]],
                         validation_triples: List[List[str]],
                         population: str,
                         prefix_of_predicted: str,
                         prediction_target: str,
                         data: PathDataset,
                         progress_bar: bool = False) -> (List[float], pd.DataFrame):
    """
    Make predictions for each term and return the score percentiles (higher is better) of test edges for each term
    """
    terms_test_edges = get_test_edges_for_term(terms,relation,test_triples, progress_bar=progress_bar)

    target_edge_index = None
    if prediction_target == 'head':
        og_col = 'tail_label'
        new_col = 'head_label'
        target_edge_index = 0
    elif prediction_target == 'tail':
        og_col = 'head_label'
        new_col = 'tail_label'
        target_edge_index = 2
    else:
        raise ValueError('prediction_target must be "head" or "tail"')
    # get the predictions for each term (hard coded as one for now)
    scores = []
    results = {'head_label':[],'relation_label':[],'tail_label':[],'rank':[],'head_degree':[],'tail_degree':[],'population':[],'rank_integer':[],'score':[]}
    for i,term in enumerate(terms):
        if 'HGNC:' not in term and 'MONDO:' not in term:
            print(f'Skipping: {term}')
            continue
        try:
            if i % (len(terms) // 50) == 0 and progress_bar:
                print(f'{i} / {len(terms)}')
        except ZeroDivisionError:
            pass
        try:
            predictions_df = predict.predict_target(model=model, head=None, relation=relation, tail=term, triples_factory=data).df
        except KeyError as e:
            print(f'KeyError: {e}')
            # raise warning with the error message
            warnings.warn(str(e))
            continue
        # add labels and remove non-gene predictions
        predictions_df = annotate(df=predictions_df,data=data,og_label=term,og_col=og_col,new_col=new_col,relation=relation, prefix_of_predicted=prefix_of_predicted,degs=degs)
        # remove rows there head and tail are the same, this is a self loop
        predictions_df = predictions_df[predictions_df['head_label'] != predictions_df['tail_label']]
        # sort the predictions by score
        predictions_df = predictions_df.sort_values(by=['score'],ascending=False)
        predictions_df['rank_integer'] = list(range(1,predictions_df.shape[0]+1))
        # assign a percentile to each prediction
        predictions_df['rank'] = predictions_df['score'].rank(pct=True)
        # assign rank as an integer to each edge
        # get the score for each edge in terms_a_test_edges
        for edge in terms_test_edges:
            if edge[1] != relation:
                continue
            # get the score for the edge, since these edges are directed in MONDO, the format should be gene,relation,term
            sub = predictions_df[predictions_df[new_col] == edge[target_edge_index]]
            # check sub shape
            if sub.shape[0] == 0:
                # this means that the edges was in the test set but not in the training set
                # I check this is the case for every instance of this in MONDO ASVD examples
                continue
            score = sub['rank'].values[0]
            scores.append(score)
            results['head_label'].append(edge[0])
            results['relation_label'].append(edge[1])
            results['tail_label'].append(edge[2])
            results['rank'].append(score)
            results['head_degree'].append(sub['head_degree'].values[0])
            results['tail_degree'].append(sub['tail_degree'].values[0])
            results['population'].append(population)
            results['rank_integer'].append(sub['rank_integer'].values[0])
            results['score'].append(sub['score'].values[0])
    results_df = pd.DataFrame(results)
    return scores, results_df

def plot_two_groups_hists(scores_a: List[float],
                 scores_b: List[float],
                 label_a: str,
                 label_b: str,
                 prefix: str):
        """
        Plot the scores for two groups as two histograms save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # make bins 0 -  1 in increments of .1
        bins = np.arange(0, 1.1, 0.1)
        axes[0].hist(scores_a,bins=bins)
        axes[0].set_title(label_a)
        axes[0].set_xlabel('Percentile')
        axes[0].set_ylabel('Frequency')
        axes[1].hist(scores_b,bins=bins)
        axes[1].set_title(label_b)
        axes[1].set_xlabel('Percentile')
        axes[1].set_ylabel('Frequency')

        # remove top and right borders
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        # match the y axis range
        max_y = max([max(axes[0].get_ylim()),max(axes[1].get_ylim())])
        axes[0].set_ylim([0,max_y])
        axes[1].set_ylim([0,max_y])

        plt.tight_layout()
        plt.savefig('{}{}_v_{}_g2p_rankings_hist.png'.format(prefix,label_a,label_b))
        plt.show()

def annotate(df: pd.DataFrame,
             data: PathDataset,
             og_label: str,
             og_col: str,
             new_col: str,
             relation: str,
             prefix_of_predicted: str,
             degs: dict) -> pd.DataFrame:
        """
        Given a dataframe output it with the following columns:
        head_label, relation_label, tail_label, score, head_degree, tail_degree
        """
        # the first column is the id of which ever entity you were trying to predict
        df[new_col] = [data.training.entity_id_to_label[x] for x in df.iloc[:,0]]
        # keep only rows where  df[new_col] contains the prefix_of_predicted
        df = df[df[new_col].str.contains(prefix_of_predicted)]
        # create a new column og_col, that is full of the og_label
        df[og_col] = og_label
        df['relation_label'] = relation
        # add the degree of the head and tail labels
        df['head_degree'] = [degs[x] if x in degs else -1 for x in df['head_label']]
        df['tail_degree'] = [degs[x] if x in degs else -1 for x in df['tail_label']]
        return df

def read_terms_from_file(filename: str) -> List[str]:
    """
    Read a list of terms from a file, one term per line
    """
    with open(filename,'r') as infile:
        terms = [x.strip() for x in infile.readlines()]
    return terms

def load_degs(filepath: str) -> dict: 
    # Load the test edges
    df: pd.DataFrame = pd.read_csv(
        filepath, sep="\t", header=None, names=['source', 'relation', 'target']
    )
    # Load these edges into a NX graph and compute the degree for each entity
    H: nx.MultiGraph = nx.from_pandas_edgelist(
        df, "source", "target", create_using=nx.MultiGraph()
    )
    degs: dict = dict(H.degree())
    return degs

def make_or_load_triples(path):
    triples = set()
    for line in open(path):
        row = line.strip().split('\t')
        triples.add((row[0], row[1], row[2]))
    return triples

def get_triples(train_path: str,validation_path: str,test_path: str) -> List[List[str]]:
    train_triples = make_or_load_triples(train_path)
    valid_triples = make_or_load_triples(validation_path)
    test_triples = make_or_load_triples(test_path)

    # remove training triples from test and validation sets
    test_triples = test_triples - train_triples
    valid_triples = valid_triples - train_triples
    # remove validation triples from test set
    test_triples = test_triples - valid_triples

    # convert to lists
    train_triples = list(train_triples)
    test_triples = list(test_triples)
    valid_triples = list(valid_triples)
    
    return train_triples, test_triples, valid_triples


def main():
    args = get_args()
    # load the model
    # Load the pretrained model
    model = torch.load(
        args.model,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    torch.set_num_threads(args.threads)

    # load terms
    group_a_terms = read_terms_from_file(args.a_terms)

    # load network to get degrees
    degs = load_degs(args.test_triples)

    # load triples
    train_triples, test_triples, validation_triples = get_triples(args.train_triples,args.validation_triples,args.test_triples)

    # write test triples to del.tsv
    with open('del.tsv','w') as outfile:
        for edge in test_triples:
            outfile.write('\t'.join(edge) + '\n')

    # load data as factory
    class TheKG(PathDataset):
        def __init__(self, **kwargs):
            super().__init__(
                training_path=args.train_triples,
                testing_path=args.test_triples,
                validation_path=args.validation_triples,
                **kwargs,
            )
    data = TheKG()

    scores_a, df_a = get_scores_for_edges(group_a_terms, 
                                          args.relation, 
                                          model, 
                                          degs, 
                                          train_triples, 
                                          test_triples, 
                                          validation_triples, 
                                          args.a_label, 
                                          prefix_of_predicted=args.prediction_prefix, 
                                          prediction_target=args.prediction_target, 
                                          data=data, 
                                          progress_bar=args.progress_bar)
    
    # write the results to a file
    df_a.to_csv(args.output,sep='\t',index=False)

if __name__ == '__main__':
    main()