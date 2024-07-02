from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# name of checkpoint for specific model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# AutoTokenizer.from_pretrainged applies predefined weights from checkpoint provided
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# classifies sentences as positive or negative
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# random list of strings for examply
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# this is the input that would be passed to the 'head' of the model, returns in pytorch format
# takes raw strings and tokenizes them to be interpretted later
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# pass our inputs to model and receive logits which are raw unnormalized scores outputted by the last layer of the model
# ** is used to unpack the dictionary 
outputs = model(**inputs)

print(f"Input:\n{inputs}")
print(f"\n\nOutput Logits Shape:\n{outputs.logits.shape}")
print(f"\n\nOutput Logits:\n{outputs.logits}")

# loss function or softmax function is applied to model output to receive the numerical percentage or probability
# that the classified sentence is part of a certain class
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(f"\n\nIntelligible Predictions:\n{predictions}")