from transformers import AutoModel
import torch

from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")

print(inputs)

inputs = {k:v.to(device) for k,v in inputs.items()}

print(inputs)

with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

print(outputs.last_hidden_state[:,0].size())

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

from datasets import load_dataset
emotions = load_dataset("emotion")

emotions.reset_format()
def tokenize(batch):
    print(type(batch))
    return tokenizer(batch["text"], padding=True, truncation=True)

print(emotions["train"][:2])

print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])\

emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True, batch_size=100)

print(emotions_hidden["train"].column_names)

import numpy as np

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
print(X_train.shape, X_valid.shape)

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_valid, y_valid))

