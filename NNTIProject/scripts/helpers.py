from copy import deepcopy
from functools import partial
from gc import collect as collect_garbage
from importlib.util import find_spec as importlib_find_spec
from itertools import product
from os import makedirs as make_dirs
from os.path import exists as path_exists
from os.path import join as path_join
from pathlib import Path
from time import time
from typing import Callable, Hashable
from warnings import warn

import h5py
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib import rcParams
from numpy import linspace as np_linspace
from numpy import ndarray
from numpy import pad as np_pad
from numpy import random as np_random
from numpy import squeeze as np_squeeze
from numpy import stack as np_stack
from sklearn.decomposition import PCA
from torch.cuda import empty_cache as cuda_empty_cache
from torch.cuda import is_available as cuda_available
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, XGLMTokenizerFast

# from sklearn.manifold import TSNE
# Using opentsne instead of sklearn.manifold.TSNE because it is faster by using the FFT method
# Install with: conda install --channel conda-forge opentsne
# See: https://github.com/pavlin-policar/openTSNE/
from openTSNE import TSNE

CACHE_DIR = "../cache/"  # Path to the cache directory
TSNE_CUDA_AVAILABLE = importlib_find_spec("tsnecuda") is not None  # Check if library tsnecuda is available

if TSNE_CUDA_AVAILABLE:
    # If `use_tsnecuda` is specified when calling `apply_tsne`, then we use the CUDA version of t-SNE
    # See: https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md
    # Install with: conda install tsnecuda -c conda-forge
    from tsnecuda import TSNE as TSNE_CUDA  # Import the CUDA version of t-SNE
else:
    warn("tsnecuda library is not available. Using opentsne instead.")


def apply_tokenizer(tokenizer: XGLMTokenizerFast, example: dict, padding: str = None):
    """Specify the tokenization function. If padding is specified, then it is used inside the tokenizer function."""
    return tokenizer(
        example["sentence"],
        padding=padding if padding else False,
        truncation=True if padding else False,
        return_tensors="pt",
    )


def add_batch_dimension(example: dict) -> dict:
    """Adds a batch dimension to the tensors. This function assumes the tensors are already in PyTorch tensors and
    simply unsqueezes them at the first dimension.
    See: https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
    """
    example["input_ids"] = example["input_ids"].unsqueeze(0)
    example["attention_mask"] = example["attention_mask"].unsqueeze(0)
    return example


def per_batch_padding_collate_fn(batch: list, tokenizer: XGLMTokenizerFast, padding: str = "longest"):
    """Dynamically pad to the longest sequence in the batch."""
    sentences = [item["sentence"] for item in batch]
    batch_padded = tokenizer(
        sentences,
        padding=padding,
        truncation=True,
        return_tensors="pt",
    )
    return batch_padded


def pad_and_stack(arrays: list[ndarray], pad_value: float = 0, pad_axis: int = 0, shift_axis: int = 0) -> ndarray:
    """Pads and stacks a list of numpy arrays along a specified axis so that they all have the same shape.
    The padding is applied to match the largest dimension size along the `pad_axis`.

    Usage:
    ```
    >>> a = np.zeros((54, 1024))
    >>> b = np.zeros((37, 1024))
    >>> pad_and_stack([a, b], pad_value=0, pad_axis=0).shape
        (2, 54, 1024)
    >>> a = np.zeros((54, 1024))
    >>> b = np.zeros((54, 1025))
    >>> pad_and_stack([a, b], pad_value=0, pad_axis=1, shift_axis=-1)
        (2, 54, 1025)
    >>> pad_and_stack([a, b], pad_value=0, pad_axis=1, shift_axis=0)
        (54, 2, 1025)
    >>> pad_and_stack([a, b], pad_value=0, pad_axis=1, shift_axis=1)
        (54, 1025, 2)
    ```

    Parameters
    ----------
    arrays : list[ndarray]
        list of numpy arrays to pad and stack
    pad_value : float, optional
        Value to use for padding the shorter arrays, by default 0
    pad_axis : int, optional
        Axis along which to pad and stack the arrays, by default 0
    shift_axis : int, optional
        Axis along which to shift the arrays, by default 0

    Returns
    -------
    np.ndarray
        A single numpy array with all input arrays padded and stacked along the specified axis. If `shift_axis` is not 0,
        then the resulting new dimension is shifted by that amount to the right.
    """
    max_size = max(array.shape[pad_axis] for array in arrays)
    padded_arrays = []

    for array in arrays:
        # Calculate padding for each dimension
        padding = [(0, 0) for _ in range(array.ndim)]
        padding[pad_axis] = (0, max_size - array.shape[pad_axis])  # Apply padding only on the pad_axis

        padded_array = np_pad(array, padding, mode="constant", constant_values=pad_value)
        padded_arrays.append(padded_array)

    # Stack along the next axis after pad_axis, to maintain separate items distinctly
    # stack_axis = pad_axis + 1 if pad_axis < arrays[0].ndim else pad_axis
    return np_stack(padded_arrays, axis=pad_axis + shift_axis)


def join_axes_of_ndarray(array: ndarray, axes_to_join: tuple[int]) -> ndarray:
    """Joins multiple axes of an array into a single axis.
    ```
    >>> a = np.zeros((2, 100, 69, 1024))
    >>> join_axes_of_array(a, (0, 1)).shape
        (200, 69, 1024)
    ```
    Parameters
    ----------
    array : ndarray
        The array to reshape.
    axes_to_join : tuple[int]
        The axes to join together.

    Returns
    -------
    ndarray
        The reshaped array with joined axes.

    Raises
    ------
    ValueError
        If the `axes_to_join` is not within the dimensions of the array or if the axes are not consecutive.
    """
    if not 0 < len(axes_to_join) <= array.ndim:
        raise ValueError("Axes to join must be within the dimensions of the array.")

    # Sort the axes and check if they are consecutive
    sorted_axes = sorted(axes_to_join)
    if sorted_axes != list(range(min(axes_to_join), max(axes_to_join) + 1)):
        raise ValueError("Axes to join must be consecutive.")

    # Calculate new shape
    new_shape = list(array.shape)
    for axis in sorted_axes[1:]:
        new_shape[sorted_axes[0]] *= new_shape[axis]
        new_shape[axis] = 1
    array = array.reshape(new_shape)

    # Now we need to move the joined axis back to the place of the first joined axis and remove all singleton dimensions
    array = np_squeeze(array, axis=tuple(range(sorted_axes[1], max(axes_to_join) + 1)))
    return array


def flatten_all_but_last(array: ndarray) -> ndarray:
    """Flattens all axes of an array except the last one.
    Usage:
    ```
    >>> a = np.zeros((2, 100, 69, 1024))
    >>> flatten_all_but_last(a).shape
        (13800, 1024)
    ```
    """
    return array.reshape(-1, array.shape[-1])


def apply_func_to_dict_arrays(
    dict_of_arrays: dict[Hashable, ndarray], func: Callable, *args, **kwargs
) -> dict[Hashable, ndarray]:
    """Applies a function to each array in a dictionary.

    Parameters
    ----------
    dict_of_arrays : dict[Hashable, ndarray]
        The dictionary with array values to apply the function to.
    func : Callable
        Function to apply to the ndarray. The first argument must be a ndarray.
    *args
        Positional arguments to pass to the generation function.
    **kwargs
        Keyword arguments to pass to the generation function.

    Returns
    -------
    dict[Hashable, ndarray]
        A dictionary with the new arrays.
    """
    return {k: func(v, *args, **kwargs) for k, v in dict_of_arrays.items()}


def random_subset_from_dict_arrays(
    dict_of_arrays: dict[Hashable, ndarray], subsample: int, random_state: int = None
) -> dict:
    """Randomly subsample the first axis of each array in a dictionary. This function is useful when we have a dictionary
    of numpy arrays and the arrays have a shape `(n_samples, n_features)` or, in general, `(n_samples, ...)`, and we need
    a random subset of the samples.

    Parameters
    ----------
    dict_of_arrays : dict[Hashable, ndarray]
        The dictionary with array values to subsample.
    subsample : int
        The number of samples to subsample from each array.
    random_state : int, optional
        The random state to use for reproducibility, by default None

    Returns
    -------
    dict
        A dictionary with the subsampled arrays.
    """
    np_random.seed(random_state)  # Set random state for reproducibility
    subsampled_dict = {}
    for key, array in dict_of_arrays.items():
        if subsample < array.shape[0]:  # Only subsample if the requested size is smaller than the current size
            # Randomly select indices without replacement
            selected_indices = np_random.choice(array.shape[0], subsample, replace=False)
            subsampled_dict[key] = array[selected_indices]
        else:
            subsampled_dict[key] = array  # If subsample size is equal or larger, return the original array
    return subsampled_dict


def split_nested_dict_by_inner_keys(nested_dict: dict) -> dict:
    """Reorganizes a nested dictionary into a dictionary of dictionaries based on the keys of the inner dictionaries,
    maintaining the original language and split keys in each new dictionary.

    Parameters
    ----------
    nested_dict : dict
        The input dictionary, where each value is another dictionary with consistent keys across all items.
        Example input:
        ```
        {
            (lang1, split1): {"tokens": ndarray, "sentences": ndarray},
            (lang2, split2): {"tokens": ndarray, "sentences": ndarray},
            ...
        }
        ```

    Returns
    -------
    dict
        A dictionary where the keys are the original inner keys from the nested dictionary ("tokens", "sentences"),
        and the values are dictionaries mapping the original outer keys (lang, split) to their respective ndarrays.
        Example output:
        ```
        {
            "tokens": {(lang1, split1): ndarray, (lang2, split2): ndarray},
            "sentences": {(lang1, split1): ndarray, (lang2, split2): ndarray}
        }
        ```
    """
    # Initialize dictionaries to store separated data based on inner keys
    reorganized_dict = {key: {} for key in next(iter(nested_dict.values())).keys()}

    # Populate the reorganized dictionary with data
    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            # Assigning ndarray directly to the respective key in the reorganized dictionary
            reorganized_dict[inner_key][outer_key] = value

    return reorganized_dict


def dict_to_hdf5(data: dict, filename: str, dst: str = None, to_cache: bool = False, subfolder: str = None) -> None:
    """Save a nested dictionary to an HDF5 file, handling up to two levels of nested dictionaries."""
    if to_cache:
        if dst is not None:
            warn("When 'to_cache' is True, 'dst' is ignored.")
        dst = CACHE_DIR
    else:
        if not dst:
            raise ValueError("dst parameter is required when 'to_cache' is False.")

    final_path = path_join(dst, subfolder) if subfolder else dst

    if not path_exists(final_path):
        make_dirs(final_path)

    with h5py.File(path_join(final_path, filename), "w") as f:

        def recursive_save(group, key, value):
            if isinstance(value, dict):
                # For dict, create a subgroup
                sub_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    recursive_save(sub_group, sub_key, sub_value)
            else:
                # Otherwise, create a dataset
                group.create_dataset(key, data=value)

        for key, value in data.items():
            recursive_save(f, key, value)


def hdf5_to_dict(
    filename: str,
    src: str = None,
    filename_is_fullpath: bool = False,
    from_cache: bool = False,
    subfolder: str = None,
    only_keys: list[Hashable] = None,
) -> dict:
    """Load a nested dictionary from an HDF5 file."""
    if from_cache:
        if src is not None:
            warn("When 'from_cache' is True, 'src' is ignored.")
        src = CACHE_DIR
    else:
        if not src and not filename_is_fullpath:
            raise ValueError("'src' parameter is required when either 'to_cache' or 'filename_is_fullpath' are False")

    # If the filename is not a full path, then join it with the source and subfolder
    if not filename_is_fullpath:
        final_path = path_join(src, subfolder, filename) if subfolder else path_join(src, filename)
    else:
        final_path = filename

    data = {}
    with h5py.File(final_path, "r") as f:

        def recursive_load(group, target_dict):
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    target_dict[key] = item[()]
                elif isinstance(item, h5py.Group):
                    sub_dict = {}
                    recursive_load(item, sub_dict)
                    target_dict[key] = sub_dict

        # Check if specific keys are requested and they exist
        keys_to_load = only_keys if only_keys else f.keys()
        for key in keys_to_load:
            if key not in f:
                warn(f"{key} not found in {filename}.")
                continue
            if isinstance(f[key], h5py.Group):
                data[key] = {}
                recursive_load(f[key], data[key])
            else:  # It's a dataset
                data[key] = f[key][()]

    return data


def files_from_pattern(directory: str, pattern: str, return_missing: bool, *args) -> list[str]:
    """Generate file names based on a pattern and lists of arguments and check their existence in a directory.

    Parameters
    ----------
    directory : str
        The directory to check for file existence.
    pattern : str
        The pattern to generate file names, with placeholders `{}` for arguments.
    return_missing : bool
        If True, return missing files; otherwise, return existing files.
    *args : list
        Variable length argument lists to fill in the pattern's placeholders.

    Returns
    -------
    list[str]
        A list of existing or missing file paths using the `{}` placeholders found in the pattern and `*args`. If
        `return_missing` is True, it returns a list of missing files; otherwise, it returns the list of existing files.
    list[tuple]
        A list of tuples representing the argument combinations for missing (or existing, based on `return_missing`) files.

    Raises
    ------
    Exception
        If the number of placeholders `{}` does not match the number of lists in `*args`.
    """
    placeholder_count = pattern.count("{}")
    if placeholder_count != len(args):
        raise Exception("The number of placeholders `{}` does not match the number of argument lists provided.")

    file_paths = []  # A list of file paths, either existing or missing depending on `return_missing = True/False`
    missing_combinations = []
    existing_combinations = []

    for combination in product(*args):
        file_path = path_join(directory, pattern.format(*combination))
        if path_exists(file_path):
            file_paths.append(file_path)
            existing_combinations.append(combination)
        else:
            file_paths.append(file_path)
            missing_combinations.append(combination)

    if return_missing:
        return file_paths, missing_combinations
    else:
        return file_paths, existing_combinations


def file_exists_in(folder: str, filename: str, recursive: bool = False, ignore_extension: bool = False) -> bool:
    """Checks if a file exists in a specified folder, with options for recursive search and ignoring the file extension.

    Parameters
    ----------
    folder : str
        The folder in which to search for the file.
    filename : str
        The name of the file to search for, with or without extension.
    recursive : bool, optional
        If True, search for the file recursively in subfolders, by default False.
    ignore_extension : bool, optional
        If True, ignore the file extension in the search, by default False.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    folder_path = Path(folder).resolve()
    if recursive:
        search_pattern = "**/*" if ignore_extension else f"**/{filename}"
    else:
        search_pattern = "*" if ignore_extension else filename

    for path in folder_path.glob(search_pattern):
        if ignore_extension:
            if path.stem == Path(filename).stem and path.is_file():
                return True
        else:
            if path.name == filename and path.is_file():
                return True
    return False


def print_dict_of_ndarrays(dictionary: dict[Hashable, ndarray], tab: str = "\t") -> None:
    """Print the shape of each array in a dictionary of numpy arrays."""
    for key, value in dictionary.items():
        print(f"{tab}Shape of array for {key}: {value.shape}")


def calculate_array_memory_size(array):
    """Calculate the memory footprint of a Numpy array."""
    # Get the number of elements in the array
    num_elements = array.size
    # Determine the size of each element in bytes
    element_size = array.itemsize
    # Calculate total memory footprint
    total_memory_bytes = num_elements * element_size
    return total_memory_bytes


def apply_pca(stacked_data: ndarray, n_components: int = 2, random_state: int = 0, verbose: bool = True) -> ndarray:
    """Apply PCA to reduce the dimensionality of the given high-dimensional array."""
    if verbose:
        print(f"\tApplying PCA... ", end="")

    start_time = time()
    pca_embedded = PCA(n_components=n_components, random_state=random_state).fit_transform(stacked_data)

    if verbose:
        print(f"Done (took {time() - start_time} s)")

    return pca_embedded


def apply_tsne(
    stacked_data: ndarray,
    n_components: int = 2,
    random_state: int = 0,
    use_tsnecuda: bool = False,
    verbose: bool = True,
) -> ndarray:
    """Apply t-SNE to reduce the dimensionality of the given high-dimensional array. NOTE that this method uses the
    `opentsne` package, which requires installing the package on the `conda` environment.

    The following parameters are set based on the number of samples in the data and the recommendations cited from
    various sources:
    * Uncertain Choices in Method Comparisons: An Illustration with t-SNE and UMAP (2023)
      See: https://epub.ub.uni-muenchen.de/107259/1/BA_Weber_Philipp.pdf
    * New guidance for using t-SNE: Alternative defaults, hyperparameter selection automation, and comparative
      evaluation (2022)
      See: https://www.sciencedirect.com/science/article/pii/S2468502X22000201
    """
    n = stacked_data.shape[0]
    learning_rate = max(200, int(n / 12))
    perplexity = 50

    if verbose:
        print(f"\tApplying t-SNE (perplexity={perplexity}, learning_rate={learning_rate})... ", end="")

    collect_garbage()  # Collect garbage
    cuda_empty_cache()  # Clear CUDA cache before running t-SNE
    tsnecuda_memoryerror = False  # Flag to check if tsnecuda failed due to a runtime error

    start_time = time()
    if use_tsnecuda and TSNE_CUDA_AVAILABLE:
        try:
            # Manual check for memory size to avoid MemoryError
            if calculate_array_memory_size(stacked_data) >= 190111744:
                raise MemoryError("The array is too large to fit in the GPU memory.")

            tsne_embedded = TSNE_CUDA(
                n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_seed=random_state
            ).fit_transform(stacked_data)
        except MemoryError as e:
            print(f"Failed to complete t-SNE with requested tsnecuda due to a MemoryError: {e}")
            print("\tOpting for opentsne instead... ", end="")
            tsnecuda_memoryerror = True
        finally:
            collect_garbage()  # Collect garbage
            cuda_empty_cache()  # Clear CUDA cache before running t-SNE

    if not use_tsnecuda or not TSNE_CUDA_AVAILABLE or tsnecuda_memoryerror:
        tsne_embedded = TSNE(
            n_components=n_components,
            n_jobs=32,
            perplexity=perplexity,
            learning_rate=learning_rate,
            initialization="pca",
            negative_gradient_method="fft",
            random_state=random_state,
        ).fit(stacked_data)

    if verbose:
        print(f"Done (took {time() - start_time} s)")

    return tsne_embedded


def scatter_plot(
    data: dict[str, ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple = (10, 8),
    cmap: str = "tab10",
    legend_size: int = 60,
    save_to_disk: bool = False,
    filename: str = None,
    show: bool = False,
    dpi: int = 300,
    ext: str = "png",
    plots_folder: str = "",
    subfolder_for_ext: bool = False,
    verbose: bool = True,
    **kwargs,
) -> None:
    """Plot 2D version of the representations."""
    # Set up figure parameters to make them look nice
    plt.rcParams["axes.formatter.use_mathtext"] = True
    rcParams["font.family"] = "cmr10"
    rcParams["axes.unicode_minus"] = False
    rcParams.update({"font.size": 14})

    # Check if the plot already exists in `self.plots_folder` and skip if it does
    if file_exists_in(plots_folder, filename, recursive=True, ignore_extension=False):
        if verbose:
            print(f"Skipped plot because {filename} already exists in {plots_folder}")
        return

    cmap = plt.get_cmap(cmap)  # Set the colormap
    colors = cmap(np_linspace(0, 1, len(data.keys())))  # Generate colors from the colormap

    plt.figure(figsize=figsize)
    for i, (key, reduced_repr) in enumerate(data.items()):
        label = key.replace("_", "$\mathrm{\_}$")
        plt.scatter(reduced_repr[:, 0], reduced_repr[:, 1], label=label, color=colors[i], **kwargs)

    legend = plt.legend()  # Generate legend
    for handle in legend.legend_handles:  # Increase the size of the circles in the legend
        handle.set_sizes([legend_size])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="major", color="k", linestyle="-", alpha=0.2)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    if save_to_disk:
        # Use the extension as a subfolder if `subfolder_for_ext` is True
        save_path = path_join(plots_folder, ext) if subfolder_for_ext else plots_folder
        make_dirs(save_path, exist_ok=True)  # Ensure the save folder exists
        save_path = path_join(save_path, f"{filename}")

        # Save the plot with the specified extension
        if ext == "svg":
            plt.savefig(save_path, format=ext)
        else:
            plt.savefig(save_path, format=ext, dpi=dpi)

        if verbose:
            print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    plt.close()


class TaskRunner:
    def __init__(
        self,
        langs: list[str],
        splits: list[str],
        model_name: str,
        dataset_name: str = "facebook/flores",
        batch_size: int = 2,
        per_batch_padding: bool = True,
        ignore_padding_token: int = -100,
        cache_dir: str = "../cache/",
        perform_early_setup: bool = True,
        verbose: bool = True,
    ) -> None:
        self.model_name = model_name
        self.str_model_name = "xglm-564M" if self.model_name == "facebook/xglm-564M" else self.model_name
        self.dataset_name = dataset_name
        self.device = "cuda" if cuda_available() else "cpu"
        self.langs = langs
        self.splits = splits
        self.per_batch_padding = per_batch_padding
        self.verbose = verbose
        self.ignore_padding_token_int = ignore_padding_token
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.setup_done = False

        # facebook/xglm-564M has 1-24 layers, gpt2 has 0-12 layers
        self.num_layers = 24 if self.model_name == "facebook/xglm-564M" else 12

        if perform_early_setup:
            self._setup()

    def _setup(self, force_setup: bool = False, warn: bool = True) -> None:
        """Setup the model and tokenizer before running the task."""
        if not self.setup_done or force_setup:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=path_join(self.cache_dir, "tokenizers")
            )

            # gpt2 does not have a padding token, so we have to add it manually
            if self.model_name == "gpt2":
                self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.unk_token})

            # Load pre-trained model from the huggingface hub.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, cache_dir=path_join(self.cache_dir, "models")
            )

            # Specify device on model and put the model into evaluation mode
            self.model = self.model.to(self.device)
            if cuda_available():
                self.model = self.model.cuda()
            self.model = self.model.eval()
            if self.verbose:
                print(f"Using device: {self.device}")

            self.setup_done = True
        elif warn:
            warn(f"Model {self.str_model_name} and tokenizer have previously being set up.")

    def _load_dataset(self, path: str, name: str = None, split: str = None):
        """Load the dataset from the specified path and name, and for the specified split, using the `datasets` library."""
        return load_dataset(
            path, name, split=split, trust_remote_code=True, cache_dir=path_join(self.cache_dir, "languages")
        )

    def load_langs(self, subset: int = -1, skip: list[tuple] = None) -> dict:
        """Load flores+ dataset for each language. The structure of the returned dictionary is as follows:
        ```
        dataset_per_lang = {
            language: {
                "dataset": {
                    split (dev/devtest): {
                        "raw": raw dataset (without tokenization),
                        "tokenized": tokenized dataset
                    }
                }
            }
        }
        ```
        Parameters
        ----------
        subset : int
            Number of examples to load for each language. If -1, all examples are loaded.
        skip : list[tuple]
            A list of tuples containing the language and split combinations to skip.

        Returns
        -------
        dict
            A dictionary with the dataset and dataloader for each language and split.
        """
        dataset = {}
        for language in self.langs:
            for split in self.splits:
                if self.verbose:
                    print(f"Loading dataset for {language} ({split})", end="... ")

                # Skip the current language and split if it is in the `skip` list
                if skip and (language, split) in skip:
                    if self.verbose:
                        print("Skipped")
                    continue

                dataset[language] = {"dataset": {}}
                dataset[language]["dataset"][split] = {}
                dataset[language]["dataset"][split]["raw"] = self._load_dataset(
                    self.dataset_name, name=language, split=split
                )

                # Subset the dataset to a certain number of sentencer per language, if subset != -1
                if subset > 0:
                    if self.verbose:
                        print(f"Selecting only {subset} examples... ", end="")

                    dataset[language]["dataset"][split]["raw"] = dataset[language]["dataset"][split]["raw"].select(
                        list(range(subset))
                    )

                if self.verbose:
                    print("Done")

        return dataset

    def tokenize_dataset(self, dataset: dict) -> dict:
        """Tokenize the dataset using a loaded pre-trained tokenizer from huggingface that goes with the specified model."""
        new_dataset = deepcopy(dataset)
        for language in dataset:
            for split in dataset[language]["dataset"]:
                # If we are to pad the whole dataset, set `padding` to longest and pass it to the tokenization function
                if not self.per_batch_padding:
                    raw_dataset = deepcopy(dataset[language]["dataset"][split]["raw"])

                    # Tokenize the dataset
                    tokenized_dataset = raw_dataset.map(
                        lambda example: apply_tokenizer(self.tokenizer, example, padding="longest"), batched=True
                    )

                    # Update the tokenized dataset with Pytorch format
                    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

                    # Apply unsqueeze operation
                    tokenized_dataset = tokenized_dataset.map(
                        lambda example: add_batch_dimension(example),
                        batched=False,  # Set batched=False to apply function to each example individually
                    )

                    # Update the new dataset with the tokenized dataset
                    new_dataset[language]["dataset"][split]["tokenized"] = tokenized_dataset
                else:  # If per batch padding is to be used, we cannot tokenize the dataset here,
                    new_dataset[language]["dataset"][split]["tokenized"] = None

        return new_dataset

    def batchify_dataset(self, unbatched_dataset: dict) -> dict:
        """Batchify the tokenized dataset using the specified tokenizer."""
        # Create a partially applied version of the collate function that includes the tokenizer. If not defined like this,
        # we get different results if the tokenizer is defined as a global variable instead of given by parameter.
        collate_fn_with_tokenizer = partial(per_batch_padding_collate_fn, tokenizer=self.tokenizer, padding="longest")

        batched_dataset = deepcopy(unbatched_dataset)
        for language in unbatched_dataset:
            batched_dataset[language]["dataloader"] = {}
            if self.verbose:
                print(f"Creating dataloaders for {language} (", end="")
            for i, split in enumerate(unbatched_dataset[language]["dataset"]):
                # If `PER_BATCH_PADDING` is False, then the padding was applied earlier and value of the key 'tokenized' is
                # not None. If it is True however, then we pass a collate_fn to dynamically apply padding per batch and
                # initialize the DataLoader
                if not self.per_batch_padding:
                    curr_dataset = unbatched_dataset[language]["dataset"][split]["tokenized"]
                else:
                    curr_dataset = unbatched_dataset[language]["dataset"][split]["raw"]

                # Set the BATCH_SIZE equal to the length of the dataset, if BATCH_SIZE == -1
                if self.batch_size == -1:
                    self.batch_size = len(curr_dataset)

                if not self.per_batch_padding:
                    batched_dataset[language]["dataloader"][split] = DataLoader(
                        curr_dataset, batch_size=self.batch_size, shuffle=False
                    )
                else:
                    batched_dataset[language]["dataloader"][split] = DataLoader(
                        curr_dataset,
                        batch_size=self.batch_size,
                        collate_fn=collate_fn_with_tokenizer,
                        shuffle=False,
                    )

                if self.verbose:
                    print(f"{split}: {len(batched_dataset[language]['dataloader'][split])}", end="")

                if self.verbose and i < len(batched_dataset[language]["dataset"]) - 1:
                    print(", ", end="")

                if self.batch_size == len(curr_dataset):
                    self.batch_size = -1

            if self.verbose:
                print(" instances)")

        return batched_dataset

    def load_langs_in_batches(self, subset: int = -1, skip: list[tuple] = None) -> dict:
        """Load the dataset, tokenize it and batchify it in one go.

        Parameters
        ----------
        subset : int, optional
            Amount of examples to load for each language. If it is -1, all examples are loaded, by default -1
        skip : list[tuple], optional
            A list of tuples containing the language and split combinations to skip, by default None

        Returns
        -------
        dict
            A dictionary containing the batchified dataset for each language and split. If the method `load_langs`
            returns {}, then this method also returns an empty dictionary, without trying to tokenize or batchify.
        """
        dataset = self.load_langs(subset=subset, skip=skip)
        # Only tokenize and batchify the dataset if the dataset is not empty
        if dataset:
            # To do this, it is obligatory to set up the model and tokenizer first
            self._setup()
            tokenized_dataset = self.tokenize_dataset(dataset)
            batched_dataset = self.batchify_dataset(tokenized_dataset)
            return batched_dataset
        else:
            return {}

    def cleanup(self):
        """Cleans up the resources by deleting model, tokenizer, and clearing CUDA cache."""
        if hasattr(self, "model") and self.model is not None:
            del self.model  # Deletes the model
            self.model = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer  # Deletes the tokenizer
            self.tokenizer = None

        collect_garbage()
        cuda_empty_cache()  # Clear CUDA cache

    def __del__(self):
        """Destructor that cleans up the resources when the instance is about to be destroyed."""
        try:
            self.cleanup()
        except Exception as e:
            warn(f"Error during cleanup: {e}")

    def run(self):
        raise NotImplementedError("This method should be implemented in the child class")
