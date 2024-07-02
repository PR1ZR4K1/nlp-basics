import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# The following lines of code are what the tokenizer does by default

# breaks the sequence of words apart
# tokens = tokenizer.tokenize(sequence)

# assigns ids to the tokenized words
# ids = tokenizer.convert_tokens_to_ids(tokens)

# transforms them into tensor object and applies an additional dimension to the array to be interpretted properly
# input_ids = torch.tensor([ids])

# finally we pass it to the model and receive the logits
# print(model(input_ids).logits)

# this is the exact same thing
inputs = model(**tokenizer(sequence, return_tensors="pt").logits)

print(model(**inputs).logits)

# ------------------------------------------------------------------ #

# example of how to use pad tokens to account for sequences of different lenght
# sequence1_ids = [[200, 200, 200]]
# sequence2_ids = [[200, 200]]
# batched_ids = [
#     [200, 200, 200],
#     [200, 200, tokenizer.pad_token_id],
# ]

# in this example sequence 2 logits will be different from the logits created for sequence 2 in the batch
# the reason is because the attention layer uses the padding to supplement context which is not good
# below is an example of attention masking which makes the layer ignore the padding tokens

# print(model(torch.tensor(sequence1_ids)).logits)
# print(model(torch.tensor(sequence2_ids)).logits)
# print(model(torch.tensor(batched_ids)).logits)


# again pad empty space to retain rectangular tensor
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# the mask identifies padding with a 0 which means it will not pay attention to it
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

# now you'll see that the logits are the same because of the attention mask

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)