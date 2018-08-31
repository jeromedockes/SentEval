import argparse
import pathlib
import pickle

import numpy as np
import logging

import senteval

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('embeddings_dir')
args = parser.parse_args()

MODEL_PATH = pathlib.Path(args.embeddings_dir / 'seed_0' / 'model.pkl')


def prepare(params, samples):
    with open(str(MODEL_PATH), 'rb') as f:
        model = pickle.load(f)
    params.model = model
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            word = word.lower()
            if word in params.model.wv.vocab:
                sentvec.append(
                    params.model.wv.vectors[params.model.wv.vocab[word].index])
        if not sentvec:
            vec = np.zeros(params.model.wv.vectors.shape[1])
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': args.data_dir, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop',
                                 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

se = senteval.engine.SE(params_senteval, batcher, prepare)
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                  'Length', 'WordContent', 'Depth', 'TopConstituents',
                  'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                  'OddManOut', 'CoordinationInversion']
results = se.eval(transfer_tasks)
print(results)
