from huggingface_hub import list_datasets
from datasets import load_dataset
emotions = load_dataset("emotion")
#print(emotions)

train_ds = emotions["train"]
print(train_ds)

print(train_ds.features)


import pandas as pd

emotions.set_format(type="pandas")

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df = emotions["train"][:]
df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

import matplotlib as plt

from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))

emotions.reset_format()
def tokenize(batch):
    print(type(batch))
    return tokenizer(batch["text"], padding=True, truncation=True)

print(emotions["train"][:2])

print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

