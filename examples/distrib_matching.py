import argparse
import pathlib
import pickle
import json
import logging

import numpy as np
from joblib import Parallel, delayed

import senteval

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('embeddings_dir')
parser.add_argument('--n_jobs', type=int, default=15)
args = parser.parse_args()


class RunScoring(object):

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.model_path = pathlib.Path(self.run_dir) / 'model.pkl'

    def prepare(self, params, samples):
        with open(str(self.model_path), 'rb') as f:
            model = pickle.load(f)
        params.model = model
        return

    def batcher(self, params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []

        for sent in batch:
            sentvec = []
            for word in sent:
                word = word.lower()
                if word in params.model.wv.vocab:
                    sentvec.append(
                        params.model.wv.vectors[
                            params.model.wv.vocab[word].index])
            if not sentvec:
                vec = np.zeros(params.model.wv.vectors.shape[1])
                sentvec.append(vec)
            sentvec = np.mean(sentvec, 0)
            embeddings.append(sentvec)

        embeddings = np.vstack(embeddings)
        return embeddings

    def compute_scores(self):
        params_senteval = {'task_path': args.data_dir,
                           'usepytorch': False, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop',
                                         'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}

        logging.basicConfig(
            format='%(asctime)s : %(message)s', level=logging.DEBUG)

        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare)
        transfer_tasks = [
            'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC']
        results = se.eval(transfer_tasks)
        print(results)
        with open(str(self.run_dir / 'sent_eval_scores.json'), 'w') as f:
            f.write(json.dumps(results))
        print(self.run_dir / 'sent_eval_scores.json')
        return results


def score_model(model_dir):
    print(model_dir)
    print('-----------')
    model_dir = pathlib.Path(model_dir)
    for run_dir in model_dir.glob('seed_*'):
        scorer = RunScoring(run_dir)
        scorer.compute_scores()


experiment_dir = pathlib.Path(args.embeddings_dir)
model_dirs = experiment_dir.glob('experiment_*')
results = Parallel(n_jobs=args.n_jobs)(
    delayed(score_model)(model_dir) for model_dir in model_dirs)


# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
#                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
#                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
#                   'Length', 'WordContent', 'Depth', 'TopConstituents',
#                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
#                   'OddManOut', 'CoordinationInversion']
