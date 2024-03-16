from functools import partial
from typing import Callable
from transformers import AutoTokenizer
from helpers.general import Cacheable
from datasets import load_dataset, Dataset

DatasetProps = tuple[str, str | None, str, Callable[[Dataset], Dataset] | None]

class DataPreparer(Cacheable):
    def __init__(self,
                tokenizer_name, 
                dataset_props: DatasetProps | dict[str][DatasetProps] = None, 
                token_size=16,
                cache_dir="cache/") -> None:
        super.__init__(cache_dir)
        self.token_size = token_size
        self.tokenizer_name = self.tokenizer_name
        self.dataset_props = dataset_props
        self.is_multiple_datasets = isinstance(dataset_props, dict)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=self.cache_dir_tokenizers)

    def get_datasets(self):
        if (self.is_multiple_datasets):
            return {k: self.get_dataset(*v) for k, v in self.dataset_props.items()}
        return self.get_dataset(*self.dataset_props)
    
    def get_dataset(self, name, sub, split, process: Callable[[Dataset], Dataset] | None):
        dataset = load_dataset(path=name, name=sub, split=split, cache_dir=self.cache_dir_datasets)
        if (process):
            dataset = process(dataset)
        return dataset
        
    # set padding token to -100 in labels
    def to_label_id(self, id):
        if (id == self.tokenizer.pad_token_id):
            return -100
        return id
    
    # preprocess sentence into token chunks (w/padding)
    def preprocess(self, tokenizer, batch):
        result = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=self.token_size
            # return_overflowing_tokens=True,
        )
        result['labels'] = list(map(self.to_label_id, result['input_ids']))

        return result
    
    def postprocess(self, dataset: Dataset):
        return dataset.remove_columns('text').with_format('torch')
    
    def generate(self):
        tokenizer = self.get_tokenizer()
        preprocess = partial(self.preprocess, tokenizer)
        if (self.is_multiple_datasets):
            return {k: self.postprocess(v.map(preprocess)) for k, v in self.get_datasets().items()}
        else :
            return self.postprocess(self.get_datasets().map(preprocess))
    
    @property
    def cache_dir_tokenizers(self):
        return self.cache_dir_sub("tokenizers")
    @property
    def cache_dir_datasets(self):
        return self.cache_dir_sub("datasets")

def process_flores(d: Dataset):
    return d.remove_columns(['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink']).rename_column("sentence", "text")

def make_small(d: Dataset, size: int):
    return d.select(range(size))

class Task3DataPreparer(DataPreparer):
    def __init__(self, dataset_props: DatasetProps | dict[str][DatasetProps] = None) -> None:
        super().__init__(
            "facebook/xglm-564M",
            dataset_props=dataset_props,
        )
    
class Task3FullDataPreparer(Task3DataPreparer):
    def __init__(self) -> None:
        super().__init__(
            {
                "train": ("Llamacha/monolingual-quechua-iic", None, "train", None),
                "test": ("facebook/flores", "quy_Latn", "devtest", process_flores)
            }
        )