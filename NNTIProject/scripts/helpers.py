import h5py
from copy import deepcopy
from typing import Hashable
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, XGLMTokenizerFast
from datasets import load_dataset
from torch.cuda import is_available as cuda_available, empty_cache as cuda_empty_cache
from torch.utils.data import DataLoader
from os.path import join as path_join, exists as path_exists
from os import makedirs as make_dirs
from numpy import (
    ndarray,
    stack as np_stack,
    pad as np_pad,
    concatenate as np_concatenate,
    zeros as np_zeros,
    mean as np_mean,
    vstack as np_vstack,
)
from warnings import warn
from itertools import product

CACHE_DIR = "../cache/"  # Path to the cache directory


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


def flatten_and_align_representations(representations: dict, pad_value: float = 0) -> ndarray:
    """
    Flatten and align the representations from different languages and batches
    to a common shape for PCA and t-SNE analysis, taking into account variable sequence lengths.

    Parameters:
    - representations (dict): A dictionary with keys as tuples (language, split) and values as numpy arrays of shape (batches, sequences, tokens, features).
    - pad_value (float, optional): Value for padding smaller arrays. Defaults to 0.

    Returns:
    - np.ndarray: A 2D numpy array suitable for PCA or t-SNE, with samples as rows and features as columns.
    """
    max_token_length = max(repr.shape[2] for repr in representations.values())
    all_pooled_reps = []

    for key, rep in representations.items():
        # Pad sequences to max_token_length
        padded_reps = [
            np_pad(
                arr,
                ((0, 0), (0, 0), (0, max_token_length - arr.shape[2]), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
            for arr in rep
        ]
        # Concatenate padded representations along the sequence axis
        concatenated_reps = np_concatenate(padded_reps, axis=1)
        # Mean pool over the token axis to get a single vector per sequence
        mean_pooled_reps = np_mean(concatenated_reps, axis=2)
        # Flatten across batches and sequences
        flattened_reps = mean_pooled_reps.reshape(-1, mean_pooled_reps.shape[-1])
        all_pooled_reps.append(flattened_reps)

    # Stack all representations vertically
    final_representation = np_vstack(all_pooled_reps)
    return final_representation


def save_hdf5(data: dict, filename: str, dst: str = None, to_cache: bool = False, subfolder: str = None) -> None:
    """Save a dictionary to an HDF5 file."""
    if to_cache:
        if dst is not None:
            warn("When 'to_cache' is True, 'dst' is ignored.")
        dst = CACHE_DIR
    else:
        if not dst:
            raise ValueError("dst parameter is required when 'to_cache' is False.")

    final_path = path_join(dst, subfolder) if subfolder else dst

    if subfolder and not path_exists(final_path):
        make_dirs(final_path)

    with h5py.File(path_join(final_path, filename), "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


def load_hdf5(
    filename: str,
    src: str = None,
    filename_is_fullpath: bool = False,
    from_cache: bool = False,
    subfolder: str = None,
    only_keys: list[Hashable] = None,
) -> dict:
    """Load a dictionary from an HDF5 file."""
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

    with h5py.File(final_path, "r") as f:
        data = {}
        for key in f.keys():
            if only_keys and key not in only_keys:
                continue
            if key not in f:
                warn(f"{key} not found in {filename}.")
            else:
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
                dataset[language]["dataset"][split]["raw"] = load_dataset(
                    self.dataset_name,
                    language,
                    split=split,
                    trust_remote_code=True,
                    cache_dir=path_join(self.cache_dir, "languages"),
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
        print(f"Attempting cleanup for {self.str_model_name}... ", end="")
        if hasattr(self, "model") and self.model is not None:
            del self.model  # Deletes the model
            self.model = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer  # Deletes the tokenizer
            self.tokenizer = None

        # Clear CUDA cache
        cuda_empty_cache()

        if self.verbose:
            print("Done")

    def __del__(self):
        """Destructor that cleans up the resources when the instance is about to be destroyed."""
        try:
            self.cleanup()
        except Exception as e:
            warn(f"Error during cleanup: {e}")

    def run(self):
        raise NotImplementedError("This method should be implemented in the child class")
