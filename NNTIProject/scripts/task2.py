from torch import inference_mode as torch_inference_mode, nonzero as torch_nonzero
from torch.cuda import empty_cache as cuda_empty_cache
from datetime import datetime
from typing import Callable
from gc import collect as collect_garbage

from helpers import (
    TaskRunner,
    flatten_all_but_last,
    dict_to_hdf5,
    files_from_pattern,
    hdf5_to_dict,
    random_subset_from_dict_arrays,
    file_exists_in,
    split_nested_dict_by_inner_keys,
    print_dict_of_ndarrays,
    apply_pca,
    apply_tsne,
    scatter_plot,
)
from os.path import join as path_join
from os.path import join as path_join
from numpy import ndarray, vstack as np_vstack, mean as np_mean
from warnings import warn
from time import time

# This is the minimal set of languages that you should analyze, feel free to experiment with additional lanuages
# available in the flores+ dataset
LANGUAGES = ["eng_Latn", "spa_Latn", "deu_Latn", "arb_Arab", "tam_Taml", "quy_Latn"]

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
        cache_dir: str = "../cache/",
        repr_folder: str = "representations",
        repr_key_pattern: str = "layer_{}",
        perform_early_setup: bool = True,
    ) -> None:
        super().__init__(langs, splits, model_name, cache_dir=cache_dir, perform_early_setup=perform_early_setup)
        self.ignore_padding_tokens_per_seq = seq_by_seq
        self.subset = subset
        self.repr_folder = path_join(self.cache_dir, repr_folder)
        self.repr_filename_pattern = (
            f"repr_{self.str_model_name}_subset_{self.subset}_seqbyseq_{self.ignore_padding_tokens_per_seq}"
            + "_{}_{}.hdf5"
        )
        self.repr_key_pattern = repr_key_pattern

    def run(self):
        # Check if representations already exist in cache directory. If so, skip computing them
        existing_reprs, existing_lang_splits = files_from_pattern(
            self.repr_folder, self.repr_filename_pattern, False, self.langs, self.splits
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

                # Filename for the token and sentence representations for the current language and split
                repr_output_filename = self.repr_filename_pattern.format(lang, split)

                if repr_output_filename in existing_reprs:
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
                            representations[repr_key] = {"tokens": [], "sentences": []}

                        # Process each sequence in the batch to obtain the token representations
                        if self.ignore_padding_tokens_per_seq:
                            for seq_idx in range(layer_states.size(0)):
                                seq_resp = layer_states[seq_idx]  # Tensor for the current sequence
                                seq_mask = attention_mask[seq_idx]  # Mask for the current sequence

                                # Extract representations excluding padding tokens using the attention mask. The shape
                                # is going to be of the form: `(sequence_length, feature_vector)`, where the sequence
                                # length is the number of tokens
                                non_padding_indices = torch_nonzero(seq_mask, as_tuple=False).squeeze()
                                non_padding_resp = seq_resp[non_padding_indices].detach().cpu().numpy()

                                # Mean-pool across the token axis to obtain a sentence representation
                                mean_pooled_resp = np_mean(non_padding_resp, axis=0)

                                # Flatten all dimensions except the last one, which is the feature dimension
                                flattened_non_padding_resp = flatten_all_but_last(non_padding_resp)
                                flattened_mean_pooled_resp = flatten_all_but_last(mean_pooled_resp)

                                # Save the token and sentence representations without padding tokens
                                representations[repr_key]["tokens"].append(flattened_non_padding_resp)
                                representations[repr_key]["sentences"].append(flattened_mean_pooled_resp)

                            # Explicitly delete tensors to free up GPU memory
                            del seq_resp, seq_mask, non_padding_indices, non_padding_resp
                        else:
                            full_repr = layer_states.detach().cpu().numpy()
                            mean_pooled_repr = np_mean(full_repr, axis=1)
                            representations[repr_key]["tokens"].append(flatten_all_but_last(full_repr))
                            representations[repr_key]["sentences"].append(flatten_all_but_last(mean_pooled_repr))

                    # Explicitly delete tensors to free up GPU memory
                    del inputs, labels, attention_mask, outputs

                # Pad and stack representations for each layer
                for layer_num in representations:
                    token_arrays, sentence_arrays = (
                        representations[layer_num]["tokens"],
                        representations[layer_num]["sentences"],
                    )

                    final_token_array_size = sum(array.shape[0] for array in token_arrays)
                    final_sentence_array_size = sum(array.shape[0] for array in sentence_arrays)

                    # Vertically stack the token and sentence representations for each layer. This should work without
                    # problems, since all vectors should be of shape (n_features,)
                    representations[layer_num]["tokens"] = np_vstack(token_arrays)
                    representations[layer_num]["sentences"] = np_vstack(sentence_arrays)

                    # For peace of mind, check the final size of the stacked arrays
                    assert final_token_array_size == representations[layer_num]["tokens"].shape[0]
                    assert final_sentence_array_size == representations[layer_num]["sentences"].shape[0]

                # Save representations to disk in cache directory
                dict_to_hdf5(representations, repr_output_filename, dst=self.repr_folder)

                if self.verbose:
                    print(f"{time() - start} s", end="")
                    if j < len(dataset[lang]["dataset"]) - 1:
                        print(", ", end="")

            if self.verbose:
                print(")")

            # After processing each language, try to free up GPU memory explicitly
            collect_garbage()
            cuda_empty_cache()


class Task2Plotter:
    def __init__(
        self,
        Task2Ran: Task2Runner,
        layer: int = 0,
        only_langs: list[str] = None,
        cache_dir: str = None,
        plots_folder: str = "plots",
    ) -> None:
        self.str_model_name = Task2Ran.str_model_name
        self.repr_folder = Task2Ran.repr_folder
        self.repr_pattern = Task2Ran.repr_filename_pattern
        self.repr_key_pattern = Task2Ran.repr_key_pattern
        self.langs = Task2Ran.langs
        self.layer = layer
        self.random_state = 0  # Random state for reproducibility used in PCA and t-SNE
        self.verbose = Task2Ran.verbose

        # Override the cache directory if it was provided, otherwise use the one from Task2Ran
        self.cache_dir = cache_dir if cache_dir else Task2Ran.cache_dir
        self.plots_folder = path_join(self.cache_dir, plots_folder)

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

        # Pattern for the filename of the plots, where the last 2 placeholders are intended for the layer number and
        # the dimensionality reduction technique, respectively
        self.token_filename_pattern = f"token_{self.str_model_name}_layer_" + "{}_{}"
        self.sentence_filename_pattern = f"sentence_{self.str_model_name}_layer_" + "{}_{}"

        # This attributes will be set in the `run` method. If it is None, the method `run` has not been called yet
        self.token_reprs_2d = None
        self.sentence_reprs_2d = None

    def _token_and_sentence_representations(self) -> tuple[dict, dict]:
        """Get the token and sentence representations to process by this runner."""
        # Load the representations for the layer for each language
        self.initial_representations = self._load_representations(self.layer)

        # Reorganize the representations so that the tokens and sentences representations are separated
        reorganized_reprs = split_nested_dict_by_inner_keys(self.initial_representations)
        token_reprs, sentence_reprs = reorganized_reprs["tokens"], reorganized_reprs["sentences"]

        if self.verbose:
            print("Initial setup of Token Representations:")
            print_dict_of_ndarrays(token_reprs)
            print("Initial setup of Sentence Representations:")
            print_dict_of_ndarrays(sentence_reprs)

        return token_reprs, sentence_reprs

    def _order_and_stack(self, repr_data: dict, subsample: int = -1) -> tuple:
        """Order and subsample the representations for each language `if subsample > 0` that are inside the given
        parameter `repr_data`. The ordered one is done by converting the dictionary of representations to a list of
        tuples, where each tuple is a language and its corresponding representation. This is necessary to stack the
        representations for each language, because dictionaries do not preserve the order of the keys. Then the
        representations are stacked in a single `ndarray` based on that ordered list of tuples.

        Parameters
        ----------
        repr_data : dict
            Dictionary containing the representations for each language, either token or sentence representations.
        subsample : int, optional
            An integer that indicates the number of samples to use for each language. If it is -1, all samples are used,
            by default -1

        Returns
        -------
        tuple
            A tuple containing the (subsampled, `if subsample > 0`) ordered representations in a single `ndarray` and
            the ordered representations as a list of tuples. If `repr_data` is empty or `None`, then the tuple will
            contain `None` for both elements.
        """
        if not repr_data:
            return None, None

        if subsample > 0:
            if self.verbose:
                print(f"Subsampling {subsample} samples for each language... ", end="")
            repr_data = random_subset_from_dict_arrays(repr_data, subsample)
            if self.verbose:
                print("Done")

        ordered_reprs = list(repr_data.items())
        stacked_array = np_vstack([array for _, array in ordered_reprs])  # Stacking the arrays vertically
        return stacked_array, ordered_reprs

    def _setup(
        self,
        token_reprs: bool = True,
        sentence_reprs: bool = True,
        for_dim_reduction: str = "PCA",
        subsample: int = -1,
        exclude_existent_plots: bool = True,
    ) -> None:
        """Sets up the `self.token_reprs` and `self.sentence_reprs` variables. It checks if a plot already exists, if
        `exclude_existent_plots == True`. If it does, then `self.token_reprs` or `self.sentence_reprs` will be None,
        correspondingly.

        Parameters
        ----------
        token_reprs : bool, optional
            True if the token representations are to be set up, by default True.
        sentence_reprs : bool, optional
            True if the sentence representations are to be set up, by default True.
        for_dim_reduction: str, optional
            Indicates which method for dimensionality reduction the `self.token_reprs` and `self.sentence_reprs` are
            being set up for. It can be either "PCA" or "t-SNE". By default, "PCA" is used.
        subsample : int, optional
            Integer that indicates the number of samples to use for each language. If it is -1, all samples are used.
        exclude_existent_plots : bool, optional
            Boolean that indicates whether to check if the plot already exists in the `self.plots_folder` before running
            the setup for either `self.token_reprs` or `self.sentence_reprs`. If it exists, the corresponding variable
            will be set to `None` to avoid recomputing. By default, True.
        """
        if for_dim_reduction not in ("PCA", "t-SNE"):
            raise ValueError(f"Unrecognized dimensionality reduction technique: {for_dim_reduction}")

        self.dim_reduction = for_dim_reduction
        self.token_reprs, self.sentence_reprs = self._token_and_sentence_representations()

        # Set up the filename for the plots that would be used in the `plot` method
        self.token_filename = self.token_filename_pattern.format(self.layer, self.dim_reduction)
        self.sentence_filename = self.sentence_filename_pattern.format(self.layer, self.dim_reduction)

        if not token_reprs:
            self.token_reprs = None
        if not sentence_reprs:
            self.sentence_reprs = None

        if exclude_existent_plots:
            if self.token_reprs and file_exists_in(
                self.plots_folder, self.token_filename, recursive=True, ignore_extension=True
            ):
                self.token_reprs = None

            if self.sentence_reprs and file_exists_in(
                self.plots_folder, self.sentence_filename, recursive=True, ignore_extension=True
            ):
                self.sentence_reprs = None

        self.stacked_token_reprs, self.ordered_token_reprs = self._order_and_stack(self.token_reprs, subsample)
        self.stacked_sentence_reprs, self.ordered_sentence_reprs = self._order_and_stack(self.sentence_reprs, subsample)

    def _apply_func_to_stacked_reprs(self, func: Callable, *args, **kwargs) -> tuple:
        """Apply the given function `func` to the token and sentence stacked representations. If they are `None`, then
        after applying the function, the corresponding output will also be `None`."""
        if self.stacked_token_reprs is not None:
            if self.verbose:
                print("On Token representations...")
            token_reprs_result = func(self.stacked_token_reprs, *args, **kwargs)
        else:
            token_reprs_result = None

        if self.stacked_sentence_reprs is not None:
            if self.verbose:
                print("On Sentence representations...")
            sentence_reprs_result = func(self.stacked_sentence_reprs, *args, **kwargs)
        else:
            sentence_reprs_result = None

        return token_reprs_result, sentence_reprs_result

    def run(
        self,
        on_token_reprs: bool = True,
        on_sentence_reprs: bool = True,
        dim_reduction: str = "PCA",
        subsample: int = -1,
        check_plot_exists: bool = False,
    ) -> None:
        """Runs the dimensionality reduction technique specified in `dim_reduction` on the representations.

        Parameters
        ----------
        dim_reduction : str, optional
            String that indicates the dimensionality reduction technique to use. It can be either "PCA" or "t-SNE".
            By default, "PCA" is used.
        subsample : int, optional
            Integer that indicates the number of samples to use for each language. If it is -1, all samples are used.
        check_plot_exists : bool, optional
            Boolean that indicates whether to check if the plot already exists in the `self.plots_folder` before running
            the dimensionality reduction technique. If it exists, the method will stop prematurely to avoid recomputing.
            If False, it will recompute everything. By default, False.
        """
        # The representations dictionary for either token and sentence should look like this after the setup:
        # For example, {('eng_Latn', 'devtest'): (17800, 1024), ('spa_Latn', 'devtest'): (13800, 1024)}
        self._setup(
            token_reprs=on_token_reprs,
            sentence_reprs=on_sentence_reprs,
            for_dim_reduction=dim_reduction,
            exclude_existent_plots=check_plot_exists,
            subsample=subsample,
        )

        if self.verbose:
            if self.stacked_token_reprs is not None:
                print(f"Final setup of ordered token representations: {self.stacked_token_reprs.shape}")
            if self.stacked_sentence_reprs is not None:
                print(f"Final setup of ordered sentence representations: {self.stacked_sentence_reprs.shape}")

        if self.dim_reduction == "PCA":
            dim_reduction_func = apply_pca
        elif self.dim_reduction == "t-SNE":
            dim_reduction_func = apply_tsne

        # Perform dimensionality reduction on the token and sentence representations
        token_reprs_2d_array, sentence_reprs_2d_array = self._apply_func_to_stacked_reprs(
            dim_reduction_func, random_state=self.random_state, verbose=self.verbose
        )

        # Separate the reduced data so we can plot it by language
        self.token_reprs_2d = self._to_repr_by_language(token_reprs_2d_array, self.ordered_token_reprs)
        self.sentence_reprs_2d = self._to_repr_by_language(sentence_reprs_2d_array, self.ordered_sentence_reprs)

        if self.verbose:
            if token_reprs_2d_array is not None:
                print(f"Shape of reduced-dimensionality token array: {token_reprs_2d_array.shape}")
            if sentence_reprs_2d_array is not None:
                print(f"Shape of reduced-dimensionality sentence array: {sentence_reprs_2d_array.shape}")

            if self.token_reprs_2d is not None:
                print("Final setup of token representations:")
                print_dict_of_ndarrays(self.token_reprs_2d)

            if self.sentence_reprs_2d is not None:
                print("Final setup of sentence representations:")
                print_dict_of_ndarrays(self.sentence_reprs_2d)

    def _load_representations(self, layer: int) -> dict:
        """Load representations for a specific layer from the class HDF5 filepaths. The structure of the loaded data is
        the following:
        ```
        {
            (lang1, split1): {"tokens": ndarray, "sentences": ndarray},
            (lang2, split1): {"tokens": ndarray, "sentences": ndarray},
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
            representations[lang_split] = hdf5_to_dict(path, filename_is_fullpath=True, only_keys=[str_layer])[
                str_layer
            ]
        return representations

    def _to_repr_by_language(self, reduced_data: ndarray, ordered_reprs: dict) -> dict:
        """Separate the reduced data back into a dictionary where the keys are the languages and the values are the
        reduced data for the corresponding language. This is a necessary method to use `plot_representations`. If
        `reduced_data` or `ordered_reprs` is `None`, then `None` is returned."""
        if reduced_data is None or ordered_reprs is None:
            return None

        repr_by_language = {}
        offset = 0
        for key, array in ordered_reprs:
            lang, _ = key
            num_samples = array.shape[0]
            repr_by_language[lang] = reduced_data[offset : offset + num_samples]
            offset += num_samples
        return repr_by_language

    def plot(
        self,
        token_repr: bool = True,
        sentence_repr: bool = True,
        figsize: tuple = (10, 8),
        dpi: int = 300,
        cmap: str = "tab10",
        legend_size: int = 60,
        save_to_disk: bool = False,
        show: bool = False,
        ext: str = "png",
        subfolder_for_ext: bool = False,
        **kwargs,
    ) -> None:
        """Plot 2D version of the representations."""
        self.token_filename += f".{ext}"
        self.sentence_filename += f".{ext}"

        data_to_plot = []
        if token_repr:
            data_to_plot.append(("Tokens", self.token_reprs_2d))
        if sentence_repr:
            data_to_plot.append(("Sentences", self.sentence_reprs_2d))

        for name, data in data_to_plot:
            if name == "Tokens":
                filename = self.token_filename
            elif name == "Sentences":
                filename = self.sentence_filename

            # Check if the plot already exists in `self.plots_folder` and skip if it does
            if file_exists_in(self.plots_folder, filename, recursive=True, ignore_extension=False):
                if self.verbose:
                    print(f"Skipping because {filename} already exists in {self.plots_folder}")
                return

            if not data:
                if self.verbose:
                    print(f"Skipping plot because the 2D representations for {name} have not been computed yet.")
                continue

            title = (
                f"{self.dim_reduction} visualization of {name} in #{self.layer} hidden layer ({self.str_model_name})"
            )
            scatter_plot(
                data,
                title,
                "Component 1",
                "Component 2",
                figsize=figsize,
                dpi=dpi,
                cmap=cmap,
                legend_size=legend_size,
                save_to_disk=save_to_disk,
                show=show,
                filename=filename,
                ext=ext,
                plots_folder=self.plots_folder,
                subfolder_for_ext=subfolder_for_ext,
                **kwargs,
            )

    def cleanup(self):
        """Explicitly delete the class attributes to free up memory."""
        vars_to_del = [
            ("token_reprs_2d", self.token_reprs_2d),
            ("sentence_reprs_2d", self.sentence_reprs_2d),
            ("ordered_token_reprs", self.ordered_token_reprs),
            ("ordered_sentence_reprs", self.ordered_sentence_reprs),
            ("token_reprs", self.token_reprs),
            ("sentence_reprs", self.sentence_reprs),
            ("stacked_sentence_reprs", self.stacked_sentence_reprs),
            ("stacked_token_reprs", self.stacked_token_reprs),
            ("initial_representations", self.initial_representations),
        ]

        for var_name, _ in vars_to_del:
            setattr(self, var_name, None)

        # Clear CUDA cache
        cuda_empty_cache()

    def __del__(self):
        """Destructor that cleans up the resources when the instance is about to be destroyed."""
        try:
            self.cleanup()
        except Exception as e:
            warn(f"Error during cleanup: {e}")


if __name__ == "__main__":
    start_time = time()
    print(f"Start time: {datetime.now()}\n")
    for dim_reduction in ["PCA", "t-SNE"]:
        for model_name in ["facebook/xglm-564M"]:
            # Create a Task2Runner instance for the current model and run it
            runner = Task2Runner(
                LANGUAGES,
                ["devtest"],  # Only analyzing the devtest split
                model_name,
                seq_by_seq=True,
                subset=200,
                cache_dir="/run/media/Camilo/Personal/Repositorios en Github/nnti/NNTIProject/cache/",
                perform_early_setup=False,
            )
            runner.run()

            # Create a Task2Plotter instance for the current model, run the current dimensionality reduction technique
            # and save each plot to disk for each layer
            for layer in range(0, runner.num_layers + 1):
                print(f"\nRunning {dim_reduction} for layer {layer} of {model_name}...\n")
                plotter = Task2Plotter(runner, layer=layer, cache_dir="../cache/", plots_folder="plots_task2")
                plotter.run(dim_reduction=dim_reduction, check_plot_exists=True)

                for ext in ["png", "svg"]:
                    plotter.plot(
                        token_repr=True,
                        sentence_repr=True,
                        cmap="Accent",
                        save_to_disk=True,
                        ext=ext,
                        subfolder_for_ext=True,
                        edgecolor="black",
                        linewidth=0.1,
                    )

                del plotter
            del runner

    print(f"\nTotal elapsed time: {time() - start_time} s")
    print(f"End time: {datetime.now()}\n")
