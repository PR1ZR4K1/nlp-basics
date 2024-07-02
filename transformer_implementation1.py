from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# tokenizer can also handle multiple sequences at the same time
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# tokenizer can pad with several options

# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
# model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
# model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

# tokenizer can also truncate by default

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
# model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
# model_inputs = tokenizer(sequences, max_length=8, truncation=True)

# finally tokenizer can return to several types of tensors
# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
# model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
# model_inputs = tokenizer(sequences, padding=True, return_tensors="np")

# another example of how to put it together from tokenizer to model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)

print(output)