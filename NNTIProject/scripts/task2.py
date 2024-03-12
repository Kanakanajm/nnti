import argparse
import torch
import h5py

import datasets
import numpy as np
import transformers

from torch import inference_mode as torch_inference_mode
from torch.cuda import empty_cache as cuda_empty_cache

from helpers import TaskRunner, save_hdf5, pad_and_stack
from time import time

# This is the minimal set of languages that you should analyze, feel free to experiment with additional lanuages
# available in the flores+ dataset
LANGUAGES = ["eng_Latn", "spa_Latn", "ita_Latn", "deu_Latn", "arb_Arab", "tel_Telu", "tam_Taml", "quy_Latn", "zho_Hans"]

########################################################
# Entry point
########################################################


class Task2Runner(TaskRunner):
    def __init__(
        self, langs: list[str], splits: list[str], model_name: str, seq_by_seq: bool = True, subset: int = 10
    ) -> None:
        super().__init__(langs, splits, model_name)
        self.seqbyseq = seq_by_seq
        self.subset = subset

    def run(self):
        dataset = self.load_langs_in_batches(self.subset)

        # Iterate over the dataset for each language and compute the cross-entropy loss per batch
        for lang in dataset:
            if self.verbose:
                print(f"Computing representations for {lang} with {self.str_model_name} (", end="")

            for j, split in enumerate(dataset[lang]["dataset"]):
                if self.verbose:
                    print(f"{split}: ", end="")

                start = time()
                dataloader = dataset[lang]["dataloader"][split]
                repr_output_file = (
                    f"repr_{self.str_model_name}_subset_{self.subset}_seqbyseq_{self.seqbyseq}_{lang}_{split}.hdf5"
                )

                representations = {}  # Dictionary to store the representations of each layer
                for _, batch in enumerate(dataloader):
                    inputs = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = (
                        batch["input_ids"].clone().to(self.device)
                    )  # Using clone() to prevent the following in-place modification

                    # Identify padding token IDs in labels and set them to self.ignore_padding_token_int
                    # See: https://nnti.sic.saarland/t/task-1-tokenization-of-data/205/12?u=cama00005
                    if self.ignore_padding_token_int and self.tokenizer.pad_token_id is not None:
                        labels[labels == self.tokenizer.pad_token_id] = self.ignore_padding_token_int

                    # torch.inference_mode() is now preferred over torch.no_grad().
                    # See: https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099/2?u=timgianitsos
                    with torch_inference_mode():
                        outputs = self.model(
                            inputs, labels=labels, attention_mask=attention_mask, output_hidden_states=True
                        )

                    # Iterate over each layer's output
                    for layer_num, layer_states in enumerate(outputs.hidden_states):
                        # Store representations for the current layer
                        if f"layer_{layer_num}" not in representations:
                            representations[f"layer_{layer_num}"] = []

                        # Process each sequence in the batch inside the current layer
                        if self.seqbyseq:
                            for seq_idx in range(layer_states.size(0)):
                                seq_resp = layer_states[seq_idx]  # Tensor for the current sequence
                                seq_mask = attention_mask[seq_idx]  # Mask for the current sequence

                                # Extract representations excluding padding tokens
                                non_padding_indices = torch.nonzero(seq_mask, as_tuple=False).squeeze()
                                non_padding_resp = seq_resp[non_padding_indices].detach().cpu().numpy()

                                representations[f"layer_{layer_num}"].append(non_padding_resp)

                            # Explicitly delete tensors to free up GPU memory
                            del seq_resp, seq_mask, non_padding_indices, non_padding_resp
                        else:
                            representations[f"layer_{layer_num}"].append(layer_states.detach().cpu().numpy())

                    # Explicitly delete tensors to free up GPU memory
                    del inputs, labels, attention_mask, outputs

                # Pad and stack representations for each layer
                for layer_num in representations:
                    layer_arrays = representations[layer_num]

                    # If seqbyseq is True, pad along the first axis. In this case, array.shape = (37, 1024) for example.
                    # Otherwise pad along the second axis, because the first axis is now the batch axis
                    # e.g, array.shape = (2, 37, 1024).
                    representations[layer_num] = pad_and_stack(
                        layer_arrays, pad_value=0, pad_axis=0 if self.seqbyseq else 1
                    )

                # Save representations to disk in cache directory
                save_hdf5(representations, self.cache_dir, repr_output_file)

                if self.verbose:
                    print(f"{time() - start} s", end="")
                    if j < len(dataset[lang]["dataset"]) - 1:
                        print(", ", end="")

            if self.verbose:
                print(")")

            # After processing each language, try to free up GPU memory explicitly
            cuda_empty_cache()


if __name__ == "__main__":
    runner = Task2Runner(LANGUAGES, ["dev", "devtest"], "facebook/xglm-564M", seq_by_seq=False, subset=200)
    runner.run()

    del runner

    runner = Task2Runner(LANGUAGES, ["dev", "devtest"], "gpt2", seq_by_seq=False, subset=200)
    runner.run()

    del runner
