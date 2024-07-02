from transformers import BertConfig, BertModel
from transformers import AutoTokenizer


# Building the config
# config = BertConfig()

# Building the model from the config
# This model would need to be trained which is dumb
# model = BertModel(config)
# basic config for the bert model
# print(config)

# Instead load a checkpoint from hugging face
# A more agnostic approach is to use AutoModel and load a checkpoint from there.
# AutoModel automatically selects the appropriate architecture based on the checkpoint provided.
model = BertModel.from_pretrained("bert-base-cased")

# autotokenizer does exactly the same thing as automodel
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"

# tokenizes a given sentence to be interpretted by our model
input = tokenizer(sequence, return_tensors='pt')
# shows how the tokenizer broke apart the given sequence
tokens = tokenizer.tokenize(sequence, return_tensors='pt')
# each token is assigned an id that corresponds to a word
ids = tokenizer.convert_tokens_to_ids(tokens)



print(model(**input))
print(tokens)

print(ids)

# decoding ids back to original string
decoded_string = tokenizer.decode(ids)

print(decoded_string)


