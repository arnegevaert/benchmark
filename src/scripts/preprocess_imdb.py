from torchtext import data
from torchtext import datasets
import json
from os import path
import torch

out_dir = path.join(path.dirname(__file__), "../../data/imdb_preproc")
data_path = path.join(path.dirname(__file__), "../../data")
TEXT = data.Field(tokenize='spacy', batch_first=True)
LABEL = data.LabelField()
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=data_path)
train_data, valid_data = train_data.split()

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_,
                 vectors_cache=path.join(data_path, "glove"))

LABEL.build_vocab(train_data)

train_examples = [t for t in train_data]
test_examples = [t for t in test_data]
valid_examples = [t for t in valid_data]

"""
with open(path.join(out_dir, "train.json"), 'w+') as f:
    for example in train_examples:
        json.dump(example, f)
        f.write('\n')

with open(path.join(out_dir, "test.json"), 'w+') as f:
    for example in test_examples:
        json.dump(example, f)
        f.write('\n')
"""
