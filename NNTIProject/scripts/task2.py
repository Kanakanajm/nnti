import argparse
import torch 
import h5py

import datasets
import numpy as np
import transformers

MODEL_NAME = "facebook/xglm-564M"
DATASET_NAME = "facebook/flores"

# this is the minimal set of languages that you should analyze
# feel free to experiment with additional lanuages available in the flores dataset
LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "arb_Arab",
    "tam_Taml",
    "quy_Latn"
]

########################################################
# Entry point
########################################################

if __name__ == "__main__":
    # TODO: your code goes here
    pass
