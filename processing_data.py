from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer

"""
our dataset.map function allows you to pass a function to be applied to each element of dataset
the reason we don't apply padding here is because the length would be set to the longest item in the dataset
instead it is better to apply padding per batch
"""

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

checkpoint = "bert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)




# print(raw_datasets["train"][0])
# print(raw_datasets["train"].features)

"""
tokenizer can ingest pairs of dicts of sentences and tokenize them together.
tokenizers for certain models will add special tokens that denote separations between
sentences and other important things necessary to retain context
this way of doing it works well only if you have enough RAM to store the entire dataset as it gets tokenized
"""

# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir="test-trainer", use_cpu=True)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)


trainer.train()

# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape)