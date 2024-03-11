import argparse
import torch 
import h5py

import datasets
import numpy as np
import transformers

from torch import inference_mode as torch_inference_mode
from torch.cuda import empty_cache as cuda_empty_cache

from helpers import TaskRunner

MODEL_NAME = "facebook/xglm-564M"
DATASET_NAME = "facebook/flores"

# this is the minimal set of languages that you should analyze
# feel free to experiment with additional lanuages available in the flores dataset
LANGUAGES = [
    "eng_Latn",
    # "spa_Latn",
    # "deu_Latn",
    # "arb_Arab",
    # "tam_Taml",
    # "quy_Latn"
]

########################################################
# Entry point
########################################################

class Task2Runner(TaskRunner):
    def __init__(self, langs, splits) -> None:
        super().__init__(langs, splits)

    def run(self):
        dataset = self.load_langs_in_batches()
            # iterate over the dataset for each language and compute the cross-entropy loss per batch
        for language in dataset:

            for j, split in enumerate(dataset[language]["dataset"]):
                print(f"{split}: ", end="")
                dataloader = dataset[language]["dataloader"][split]

                for _, batch in enumerate(dataloader):
                    inputs = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = (
                        batch["input_ids"].clone().to(self.device)
                    )  # Using clone() to prevent the following in-place modification

                    # Identify padding token IDs in labels and set them to -100
                    # See: https://nnti.sic.saarland/t/task-1-tokenization-of-data/205/12?u=cama00005
                    if self.ignore_padding_token_int and self.tokenizer.pad_token_id is not None:
                        labels[labels == self.tokenizer.pad_token_id] = self.ignore_padding_token_int

                    # torch.inference_mode() is now preferred over torch.no_grad().
                    # See: https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099/2?u=timgianitsos
                    with torch_inference_mode():
                        outputs = self.model(inputs, labels=labels, attention_mask=attention_mask, output_hidden_states=True)
                        hidden_states = outputs.hidden_states

                    # Explicitly delete tensors to free up GPU memory
                    del inputs, labels, attention_mask, outputs, hidden_states

            # After processing each language, try to free up memory explicitly
            cuda_empty_cache()  # Frees unused memory so it can be used by other tensors

if __name__ == "__main__":
    # TODO: your code goes here
    runner = Task2Runner(LANGUAGES, ["dev"])
    runner.run()
