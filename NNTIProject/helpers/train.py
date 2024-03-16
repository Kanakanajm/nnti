from transformers import TrainingArguments, Trainer
# pass "wandb" to the 'report_to' parameter to turn on wandb logging

class DefaultTrainingArguments(TrainingArguments):
    def __init__(self) -> None:
        super().__init__(
            output_dir="fine-tuned-xglm-564M",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
        )

class WandbTrainingArguments(TrainingArguments):
    def __init__():
        super().__init__(
        output_dir="fine-tuned-xglm-564M",
        report_to="wandb",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        evaluation_strategy="steps",
        eval_steps=20,
        logging_steps=5, 
        max_steps = 100,
        save_steps = 100
    )