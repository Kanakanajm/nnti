from torch import inference_mode as torch_inference_mode, nonzero as torch_nonzero
from torch.cuda import empty_cache as cuda_empty_cache

from helpers import (
    TaskRunner,
    save_hdf5,
    pad_and_stack,
    files_from_pattern,
    load_hdf5,
    print_dict_of_ndarrays,
    apply_func_to_dict_arrays,
    join_axes_of_ndarray,
)
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from os.path import join as path_join
from numpy import ndarray
from warnings import warn
from time import time

# This is the minimal set of languages that you should analyze, feel free to experiment with additional lanuages
# available in the flores+ dataset
LANGUAGES = ["eng_Latn", "spa_Latn", "ita_Latn", "deu_Latn", "arb_Arab", "tel_Telu", "tam_Taml", "quy_Latn", "zho_Hans"]
SPLITS = ["devtest"]
MODELS = ["facebook/xglm-564M", "gpt2"]

########################################################
# Entry point
########################################################


class Task2Runner(TaskRunner):
    def __init__(
        self,
        langs: list[str],
        splits: list[str],
        model_name: str,
        seq_by_seq: bool = True,
        subset: int = 10,
        repr_folder: str = "representations",
        repr_key_pattern: str = "layer_{}",
        perform_early_setup: bool = True,
    ) -> None:
        super().__init__(langs, splits, model_name, perform_early_setup=perform_early_setup)
        self.seqbyseq = seq_by_seq
        self.subset = subset
        self.repr_folder = path_join(self.cache_dir, repr_folder)
        self.repr_pattern = f"repr_{self.str_model_name}_subset_{self.subset}_seqbyseq_{self.seqbyseq}" + "_{}_{}.hdf5"
        self.repr_key_pattern = repr_key_pattern

    def run(self):
        # Check if representations already exist in cache directory. If so, skip computing them
        existing_reprs, existing_lang_splits = files_from_pattern(
            self.repr_folder, self.repr_pattern, False, self.langs, self.splits
        )

        dataset = self.load_langs_in_batches(self.subset, skip=existing_lang_splits)

        # Skip if dataset is empty, which means that the representations for all languages and splits already exist
        if not dataset:
            return

        # Perform setup for model and tokenizer, without warning because it could have already being set up inside
        # the method `load_langs_in_batches`
        super()._setup(warn=False)

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

                if repr_output_file in existing_reprs:
                    if self.verbose:
                        print(f"skipped: already exists")
                    continue

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
                        repr_key = self.repr_key_pattern.format(layer_num)
                        # Store representations for the current layer
                        if repr_key not in representations:
                            representations[repr_key] = []

                        # Process each sequence in the batch inside the current layer
                        if self.seqbyseq:
                            for seq_idx in range(layer_states.size(0)):
                                seq_resp = layer_states[seq_idx]  # Tensor for the current sequence
                                seq_mask = attention_mask[seq_idx]  # Mask for the current sequence

                                # Extract representations excluding padding tokens
                                non_padding_indices = torch_nonzero(seq_mask, as_tuple=False).squeeze()
                                non_padding_resp = seq_resp[non_padding_indices].detach().cpu().numpy()

                                representations[repr_key].append(non_padding_resp)

                            # Explicitly delete tensors to free up GPU memory
                            del seq_resp, seq_mask, non_padding_indices, non_padding_resp
                        else:
                            representations[repr_key].append(layer_states.detach().cpu().numpy())

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
                save_hdf5(representations, repr_output_file, dst=self.repr_folder)

                if self.verbose:
                    print(f"{time() - start} s", end="")
                    if j < len(dataset[lang]["dataset"]) - 1:
                        print(", ", end="")

            if self.verbose:
                print(")")

            # After processing each language, try to free up GPU memory explicitly
            cuda_empty_cache()


class Task2Plotter:
    def __init__(self, Task2Ran: Task2Runner, dim_reduction: str, layer: int = 0, only_langs: list[str] = None) -> None:
        self.repr_folder = Task2Ran.repr_folder
        self.repr_pattern = Task2Ran.repr_pattern
        self.repr_key_pattern = Task2Ran.repr_key_pattern
        self.langs = Task2Ran.langs
        self.layer = layer
        self.dim_reduction = dim_reduction
        if self.dim_reduction not in ("PCA", "t-SNE"):
            raise ValueError(f"Unrecognized dimensionality reduction technique: {dim_reduction}")

        # Pick the only_langs subset if it was provided
        if only_langs:
            langs_set = set(self.langs)
            keep_set = set(only_langs)
            # Check for elements in only_langs not present in self.langs
            not_present = keep_set - langs_set
            if not_present:
                warn(f"The languages in 'only_langs' are not present in 'Task2Ran.langs': {', '.join(not_present)}")
            # Exclude elements in exclude_langs from self.langs
            self.langs = list(keep_set)

        self.splits = Task2Ran.splits

        # This will be a tuple where the first element is a list of filepaths and the second element is a list of
        # language and split tuples for each corresponding filepath (order is preserved)
        self.hdf5_files, self.lang_splits = files_from_pattern(
            self.repr_folder, self.repr_pattern, False, self.langs, self.splits
        )

        # Get the representations for the specified layer for each language
        # For example, {('eng_Latn', 'devtest'): (2, 100, 69, 1024), ('spa_Latn', 'devtest'): (2, 100, 89, 1024)}
        self.representations = self.load_representations(self.layer)

        # Apply the function `join_axes_of_ndarray` to the representations to join the first and second axes, which
        # are the batch and sequence axes, respectively. Now, the representations would look like:
        # For example, {('eng_Latn', 'devtest'): (200, 69, 1024), ('spa_Latn', 'devtest'): (200, 89, 1024)}
        self.representations = apply_func_to_dict_arrays(self.representations, join_axes_of_ndarray, (0, 1))

        # Stack the representations for each language so that they can be used for dimensionality reduction
        reprs_to_pad = [value for _, value in self.representations.items()]

        # Use the pad_and_stack function to equalize the third dimension (sequence length)
        # (batch size, number of sequences, sequence length, feature size)
        padded_reprs = pad_and_stack(reprs_to_pad, pad_axis=1, shift_axis=-1)

        # Now each sequence in 'padded_arrays' has the same length along axis=2
        # Next step is to flatten the arrays and concatenate all the data
        flattened_reprs = padded_reprs.reshape(padded_reprs.shape[0] * padded_reprs.shape[1], -1)

        if Task2Ran.verbose:
            print("Initial setup:")
            print_dict_of_ndarrays(self.representations)
            print(f"Shape of flattened array: {flattened_reprs.shape}")
            print(f"Applying {self.dim_reduction}... ", end="")

        # Perform dimensionality reduction
        if self.dim_reduction == "PCA":
            reduced_repr = self.apply_pca(flattened_reprs)
        elif self.dim_reduction == "t-SNE":
            reduced_repr = self.apply_tsne(flattened_reprs)

        # Separate the reduced data so we can plot it by language
        self.reduced_reprs = self.to_repr_by_language(reduced_repr)

        if Task2Ran.verbose:
            print("Done")
            print(f"Shape of reduced array: {reduced_repr.shape}")
            print("Final setup:")
            print_dict_of_ndarrays(self.reduced_reprs)

    def load_representations(self, layer: int) -> dict:
        """Load representations for a specific layer from the class HDF5 filepaths. The structure of the loaded data is
        the following:
        ```
        {
            (lang1, split1): ndarray,
            (lang2, split1): ndarray,
            ...
        }
        ```
        Parameters
        ----------
        layer : int
            Integer that indicates the layer to load the representations from.

        Returns
        -------
        dict
            A dictionary containing the loaded representations for each language.
        """
        str_layer = self.repr_key_pattern.format(layer)
        representations = {}
        for path, lang_split in zip(self.hdf5_files, self.lang_splits):
            representations[lang_split] = load_hdf5(path, filename_is_fullpath=True, only_keys=[str_layer])[str_layer]
        return representations

    def apply_pca(self, stacked_data: ndarray, n_components: int = 2) -> ndarray:
        """Apply PCA to reduce the dimensionality of the given high-dimensional array."""
        pca = PCA(n_components=n_components)
        return pca.fit_transform(stacked_data)

    def apply_tsne(self, stacked_data: ndarray, n_components: int = 2, random_state: int = 42) -> ndarray:
        """Apply t-SNE to reduce the dimensionality of the given high-dimensional array."""
        tsne = TSNE(n_components=n_components, random_state=random_state)
        return tsne.fit_transform(stacked_data)

    def to_repr_by_language(self, reduced_data: ndarray) -> dict:
        """Separate the reduced data back into a dictionary where the keys are the languages and the values are the
        reduced data for the corresponding language. This is a necessary method to fit into `plot_representations`."""
        repr_by_language = {}
        offset = 0
        for key, array in self.representations.items():
            lang, _ = key
            num_samples = (
                array.shape[0] * array.shape[1]
            )  # number of sentences times the number of samples per language
            repr_by_language[lang] = reduced_data[offset : offset + num_samples]
            offset += num_samples
        return repr_by_language

    def plot_representations(self) -> None:
        """Plot 2D version of the representations."""
        title = f"{self.dim_reduction} visualization of #{self.layer} hidden layer"
        plt.figure(figsize=(10, 8))
        for lang, reduced_repr in self.reduced_reprs.items():
            plt.scatter(
                reduced_repr[:, 0], reduced_repr[:, 1], label=lang, c="lightblue", edgecolor="black", s=50, alpha=0.6
            )
        plt.legend()
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Execute these lines to generate representations
    for model_name in MODELS:
        runner = Task2Runner(LANGUAGES, SPLITS, model_name, seq_by_seq=False, subset=200, perform_early_setup=False)
        runner.run()
        runner.cleanup()
        plotter = Task2Plotter(runner, dim_reduction="PCA", layer=24, only_langs=["eng_Latn", "spa_Latn"])
        plotter.plot_representations()
        import sys

        sys.exit(0)
