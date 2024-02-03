import os
from typing import List
from gensim import utils
import gensim.models

DATA_ROOT_PATH = "./data/aclImdb"
SPLITS = ["train", "test"]
SENTIMENTS = ["pos", "neg"]

MODEL_SAVE_SUBPATH = 'tmp/gensim-data/gensim-model-w2v-imdb.tmp'
MODEL_SAVE_PATH = os.path.join(os.environ['VIRTUAL_ENV'], MODEL_SAVE_SUBPATH)



class IMDBCorpus:
    def __init__(self, data_paths: List[str]):
        self.data_paths = data_paths

    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        for path in self.data_paths:
            for filename in os.listdir(path):
                if filename.endswith(".txt"):
                    with open(os.path.join(path, filename), 'r') as file:
                        yield utils.simple_preprocess(file.read())

class IMDBLabels:
    def __init__(self, data_paths: List[str]):
        self.data_paths = data_paths

    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        for path in self.data_paths:
            for filename in os.listdir(path):
                if filename.endswith(".txt"):
                    yield int(os.path.splitext(filename)[0].split('_')[1])

class IMDBTrainLabels(IMDBLabels):
    def __init__(self):
        super().__init__([os.path.join(DATA_ROOT_PATH, "train", sent) for sent in SENTIMENTS])

class IMDBTestLabels(IMDBLabels):
    def __init__(self):
        super().__init__([os.path.join(DATA_ROOT_PATH, "test", sent) for sent in SENTIMENTS])                   

class IMDBTrainCorpus(IMDBCorpus):
    def __init__(self):
        super().__init__([os.path.join(DATA_ROOT_PATH, "train", sent) for sent in SENTIMENTS])

class IMDBTestCorpus(IMDBCorpus):
    def __init__(self):
        super().__init__([os.path.join(DATA_ROOT_PATH, "test", sent) for sent in SENTIMENTS])

class IMDBCompleteCorpus(IMDBCorpus):
    def __init__(self):
        super().__init__([os.path.join(DATA_ROOT_PATH, split, sent) for split in SPLITS for sent in SENTIMENTS])

def train():
    corpus = IMDBCompleteCorpus()
    model = gensim.models.Word2Vec(sentences=corpus)
    return model

def load():
    return gensim.models.Word2Vec.load(MODEL_SAVE_PATH)



