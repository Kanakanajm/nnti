from copy import deepcopy
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, XGLMForCausalLM, XGLMTokenizerFast
from datasets import load_dataset
from torch.cuda import is_available as cuda_available
from torch.utils.data import DataLoader
from os.path import join as path_join

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

class TaskRunner:
    def __init__(self, langs, splits) -> None:
        self.model_name = "facebook/xglm-564M"
        self.dataset_name = "facebook/flores"
        self.device = "cuda" if cuda_available() else "cpu"
        self.langs = langs
        self.splits = splits
        self.per_batch_padding = True
        self.verbose = True
        self.ignore_padding_token_int = -100
        self.cache_dir = "cache"
        self.batch_size = 2

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=path_join(self.cache_dir, "tokenizers"))
        
        """Load pre-trained model from the huggingface hub."""
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=path_join(self.cache_dir,"models"))

        # specify device on model and put the model into evaluation mode
        self.model = self.model.to(self.device)
        if cuda_available():
            self.model = self.model.cuda()
        self.model = self.model.eval()
        if self.verbose:
            print(f"Using device: {self.device}")

    """
    load flores data for each language
    structure:
    dataset_per_lang = {
      language: {
          "dataset": {
               split (dev/devtest): {
                   "raw": raw dataset (without tokenization),
                   "tokenized": tokenized dataset
               }
          },
          "dataloader": {
               split (dev/devtest): dataloader for the split
          }
      }
    }
    """
    def load_langs(self) -> dict:
        dataset = {}
        for language in self.langs:
            print(f"Loading dataset for {language}", end="... ")

            # add a dataloader key set to None, they are defined in the cell tagged @dataloader-creation
            dataset[language] = {"dataset": {}, "dataloader": None}

            for split in self.splits:
                dataset[language]["dataset"][split] = {}
                dataset[language]["dataset"][split]["raw"] = load_dataset(
                    self.dataset_name,
                    language,
                    split=split,
                    trust_remote_code=True,
                    cache_dir=path_join(self.cache_dir,"languages"),
                )

            print("done")
        return dataset
    
    def tokenize_dataset(self, dataset: dict) -> dict:
        """Tokenize the dataset using a loaded pre-trained tokenizer from huggingface that goes with the specified model."""

        # gpt2 does not have a padding token, so we have to add it manually
        if self.model_name == "gpt2":
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.unk_token})

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
    
    def load_langs_in_batches(self):
        dataset = self.load_langs()
        tokenized_dataset = self.tokenize_dataset(dataset)
        batched_dataset = self.batchify_dataset(tokenized_dataset)
        return batched_dataset
    
    def run(self):
        raise NotImplementedError("This method should be implemented in the child class")