import argparse
import torch

import datasets
import numpy as np
import transformers
from helpers import TaskRunner
import wandb

MODEL_NAME = "facebook/xglm-564M"
DATASET_FOR_FINETUNING = "hackathon-pln-es/spanish-to-quechua"

########################################################
# Entry point
########################################################

# This is the minimal set of languages that you should analyze, feel free to experiment with additional lanuages
# available in the flores+ dataset
LANGUAGES = ["eng_Latn", "spa_Latn", "deu_Latn", "arb_Arab", "tam_Taml", "quy_Latn"]


class Task3Runner(TaskRunner):
    def __init__(
        self,
        langs: list[str],
        splits: list[str],
        model_name: str,
        dataset_for_finetuning: str,
        cache_dir: str = "../cache/",
        perform_early_setup: bool = True,
    ) -> None:
        super().__init__(langs, splits, model_name, cache_dir=cache_dir, perform_early_setup=perform_early_setup)
        self.dataset_for_finetuning = self._load_dataset(dataset_for_finetuning)
        print(self.dataset_for_finetuning["train"])


if __name__ == "__main__":
    runner = Task3Runner(
        LANGUAGES,
        ["devtest"],
        MODEL_NAME,
        DATASET_FOR_FINETUNING,
        cache_dir="/run/media/Camilo/Personal/Repositorios en Github/nnti/NNTIProject/cache/",
        perform_early_setup=False,
    )
