from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))

# 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

from transformers import Trainer, TrainingArguments

from datasets import load_dataset
emotions = load_dataset("emotion")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    print(type(batch))
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])\

batch_size=1

logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=True,
                                  log_level="error")


from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

trainer.train()