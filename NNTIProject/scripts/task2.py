from torch import inference_mode as torch_inference_mode, nonzero as torch_nonzero
from torch.cuda import empty_cache as cuda_empty_cache
from datetime import datetime

from helpers import (
    TaskRunner,
    flatten_all_but_last,
    save_hdf5,
    pad_and_stack,
    files_from_pattern,
    load_hdf5,
    print_dict_of_ndarrays,
    apply_func_to_dict_arrays,
    flatten_all_but_last,
    random_subset_from_dict_arrays,
    check_file_existence,
)
from matplotlib import pyplot as plt, rcParams

# from sklearn.manifold import TSNE
# Using tsnecuda instead of sklearn.manifold.TSNE because it is faster and can be run on the GPU
# See: https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md
# Install with: conda install tsnecuda -c conda-forge
from tsnecuda import TSNE
from sklearn.decomposition import PCA

from os.path import join as path_join
from os import makedirs as make_dirs
from os.path import join as path_join
from numpy import ndarray, vstack as np_vstack, linspace as np_linspace
from warnings import warn
from time import time

# This is the minimal set of languages that you should analyze, feel free to experiment with additional lanuages
# available in the flores+ dataset
LANGUAGES = ["eng_Latn", "spa_Latn", "deu_Latn", "arb_Arab", "tam_Taml", "quy_Latn"]
SPLITS = ["devtest"]
MODELS = ["facebook/xglm-564M", "gpt2"]

# Set up figure parameters to make them look nice
plt.rcParams["axes.formatter.use_mathtext"] = True
rcParams["font.family"] = "cmr10"
rcParams["axes.unicode_minus"] = False
rcParams.update({"font.size": 14})

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
        self.repr_pattern = Task2Ran.repr_pattern
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
        self.plot_filename_pattern = f"{self.str_model_name}_layer_" + "{}_{}"

        # This attribute will be set in the `run` method. If it is None, the method `run` has not been called yet
        self.reduced_reprs = None

    def run(self, dim_reduction: str = "PCA", subsample: int = -1, check_plot_exists: bool = False) -> None:
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
        if dim_reduction not in ("PCA", "t-SNE"):
            raise ValueError(f"Unrecognized dimensionality reduction technique: {dim_reduction}")

        self.dim_reduction = dim_reduction

        # Check if the plot already exists if `check_plot_existence_in` is provided, which is intended to be a folder
        if check_plot_exists:
            plot_filename = self.plot_filename_pattern.format(self.layer, self.dim_reduction)
            if check_file_existence(self.plots_folder, plot_filename, recursive=True, ignore_extension=True):
                return

        # Get the representations for the specified layer for each language
        # For example, {('eng_Latn', 'devtest'): (2, 100, 69, 1024), ('spa_Latn', 'devtest'): (2, 100, 89, 1024)}
        self.initial_representations = self.load_representations(self.layer)

        if self.verbose:
            print("Initial setup:")
            print_dict_of_ndarrays(self.initial_representations)

        # Apply the function `flatten_all_but_last` to the representations to join the first, second and third axes,
        # which are the batch, number of sequences and sequence length axes, respectively. Now, the representations
        # would look like: `(n_samples, n_features)`.
        # For example, {('eng_Latn', 'devtest'): (17800, 1024), ('spa_Latn', 'devtest'): (13800, 1024)}
        self.flattened_representations = apply_func_to_dict_arrays(self.initial_representations, flatten_all_but_last)

        # Subsample the representations for each language if `subsample` is greater than 0
        if subsample > 0:
            if self.verbose:
                print(f"Subsampling {subsample} samples for each language... ", end="")
            self.flattened_representations = random_subset_from_dict_arrays(self.flattened_representations, subsample)
            if self.verbose:
                print("Done")

        # Convert the dictionary of representations to a list of tuples, where each tuple is a language and its
        # corresponding representation. This is necessary to stack the representations for each language, because
        # dictionaries do not preserve the order of the keys.
        self.ordered_reprs = list(self.flattened_representations.items())

        # Stack the representations for each language so that they can be used for dimensionality reduction.
        # For example, the stacked representations would have a shape of: (31600, 1024) for the example above
        reprs_to_stack = [value for _, value in self.ordered_reprs]
        stacked_reprs = np_vstack(reprs_to_stack)

        if self.verbose:
            print(f"Intermediate setup (before {self.dim_reduction}):")
            print_dict_of_ndarrays(self.flattened_representations)
            print(f"Shape of flattened array: {stacked_reprs.shape}")

        # Perform dimensionality reduction
        start_time = time()
        if self.dim_reduction == "PCA":
            reduced_repr = self.apply_pca(stacked_reprs, random_state=self.random_state)
        elif self.dim_reduction == "t-SNE":
            reduced_repr = self.apply_tsne(stacked_reprs, random_state=self.random_state)
        elapsed_time = time() - start_time

        # Separate the reduced data so we can plot it by language
        self.reduced_reprs = self.to_repr_by_language(reduced_repr)

        if self.verbose:
            print(" (took {:.2f} s)".format(elapsed_time))
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

    def apply_pca(self, stacked_data: ndarray, n_components: int = 2, random_state: int = 0) -> ndarray:
        """Apply PCA to reduce the dimensionality of the given high-dimensional array."""
        if self.verbose:
            print(f"Applying PCA... ", end="")

        embedded = PCA(n_components=n_components, random_state=random_state).fit_transform(stacked_data)

        if self.verbose:
            print("Done", end="")

        return embedded

    def apply_tsne(self, stacked_data: ndarray, n_components: int = 2, random_state: int = 0) -> ndarray:
        """Apply t-SNE to reduce the dimensionality of the given high-dimensional array. NOTE that this method uses the
        `tsnecuda` package, which requires a GPU and of course, installing the package on the `conda` environment."""

        # The following parameters are set based on the number of samples in the data and the recommendations cited from
        # various sources:
        # * Uncertain Choices in Method Comparisons: An Illustration with t-SNE and UMAP (2023)
        #   See: https://epub.ub.uni-muenchen.de/107259/1/BA_Weber_Philipp.pdf
        # * New guidance for using t-SNE: Alternative defaults, hyperparameter selection automation, and comparative
        #   evaluation (2022)
        #   See: https://www.sciencedirect.com/science/article/pii/S2468502X22000201
        n = stacked_data.shape[0]
        learning_rate = max(200, int(n / 12))
        perplexity = max(30, int(n / 100))

        if self.verbose:
            print(f"Applying t-SNE (perplexity={perplexity}, learning_rate={learning_rate})... ", end="")

        embedded = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            verbose=False,
            random_seed=random_state,
        ).fit_transform(stacked_data)

        if self.verbose:
            print("Done", end="")

        return embedded

    def to_repr_by_language(self, reduced_data: ndarray) -> dict:
        """Separate the reduced data back into a dictionary where the keys are the languages and the values are the
        reduced data for the corresponding language. This is a necessary method to fit into `plot_representations`.
        This function assumes that the `self.flattened_representations` dictionary has already been flattened for each
        language, so that the values are ndarrays of shape `(n_samples, n_features)`, and this dictionary has been
        dumped into `self.ordered_reprs` to preserve ordering as a list of tuples.
        """
        repr_by_language = {}
        offset = 0
        for key, array in self.ordered_reprs:
            lang, _ = key
            num_samples = array.shape[0]
            repr_by_language[lang] = reduced_data[offset : offset + num_samples]
            offset += num_samples
        return repr_by_language

    def plot_representations(
        self,
        figsize: tuple = (10, 8),
        dpi: int = 300,
        cmap: str = "tab10",
        legend_size: int = 60,
        save_to_disk: bool = False,
        show: bool = False,
        extension: str = "png",
        **kwargs,
    ) -> None:
        """Plot 2D version of the representations."""
        filename = self.plot_filename_pattern.format(self.layer, self.dim_reduction) + f".{extension}"

        # Check if the plot already exists in `self.plots_folder` and skip if it does
        if check_file_existence(self.plots_folder, filename, recursive=True, ignore_extension=False):
            if self.verbose:
                print(f"Skipping because {filename} already exists in {self.plots_folder}")
            return

        if not self.reduced_reprs:
            raise ValueError("The reduced representations have not been computed yet. Call `run` first.")

        cmap = plt.get_cmap(cmap)  # Set the colormap
        colors = cmap(np_linspace(0, 1, len(self.reduced_reprs.keys())))  # Generate colors from the colormap

        title = f"{self.dim_reduction} visualization of #{self.layer} hidden layer ({self.str_model_name})"
        plt.figure(figsize=figsize)
        for i, (lang, reduced_repr) in enumerate(self.reduced_reprs.items()):
            label = lang.replace("_", "$\mathrm{\_}$")
            plt.scatter(reduced_repr[:, 0], reduced_repr[:, 1], label=label, color=colors[i], **kwargs)

        legend = plt.legend()  # Generate legend
        for handle in legend.legend_handles:  # Increase the size of the circles in the legend
            handle.set_sizes([legend_size])

        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, which="major", color="k", linestyle="-", alpha=0.2)
        plt.gca().set_axisbelow(True)
        plt.tight_layout()

        if save_to_disk:
            make_dirs(self.plots_folder, exist_ok=True)  # Ensure the save folder exists
            # Save the plot with specified extension
            save_path = path_join(self.plots_folder, f"{filename}.{extension}")
            if extension == "svg":
                plt.savefig(save_path, format=extension)
            else:
                plt.savefig(save_path, format=extension, dpi=dpi)
            if self.verbose:
                print(f"Plot saved to {save_path}")

        if show:
            plt.show()
        plt.close()

    def cleanup(self):
        """Explicitly delete the class attributes to free up memory."""
        if hasattr(self, "initial_representations") and self.initial_representations is not None:
            del self.initial_representations
            self.model = None

        if hasattr(self, "flattened_representations") and self.flattened_representations is not None:
            del self.flattened_representations
            self.flattened_representations = None

        if hasattr(self, "reduced_reprs") and self.reduced_reprs is not None:
            del self.reduced_reprs
            self.reduced_reprs = None

        if hasattr(self, "ordered_reprs") and self.ordered_reprs is not None:
            del self.ordered_reprs
            self.ordered_reprs = None

        # Clear CUDA cache
        cuda_empty_cache()

        del self.hdf5_files


if __name__ == "__main__":
    start_time = time()
    print(f"Start time: {datetime.now()}\n")
    for dim_reduction in ["PCA", "t-SNE"]:
        for model_name in MODELS:
            # Create a Task2Runner instance for the current model and run it
            runner = Task2Runner(
                LANGUAGES,
                SPLITS,
                model_name,
                seq_by_seq=False,
                subset=200,
                cache_dir="/run/media/Camilo/Personal/Repositorios en Github/nnti/NNTIProject/cache",
                perform_early_setup=False,
            )
            runner.run()
            runner.cleanup()

            # Create a Task2Plotter instance for the current model, run the current dimensionality reduction technique
            # and save each plot to disk for each layer
            for layer in range(0, runner.num_layers + 1):
                print(f"\nRunning {dim_reduction} for layer {layer} of {model_name}...\n")
                plotter = Task2Plotter(runner, layer=layer, cache_dir="../cache/")
                plotter.run(dim_reduction=dim_reduction, check_plot_exists=True)
                plotter.plot_representations(
                    cmap="Accent",
                    save_to_disk=True,
                    extension="png",
                    edgecolor="black",
                    linewidth=0.1,
                )
                plotter.plot_representations(
                    cmap="Accent",
                    save_to_disk=True,
                    extension="svg",
                    edgecolor="black",
                    linewidth=0.1,
                )
                plotter.cleanup()
                del plotter
            del runner

    print(f"\nTotal elapsed time: {time() - start_time} s")
    print(f"End time: {datetime.now()}\n")
