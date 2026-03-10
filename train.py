#import neceessary libraries
import pandas as pd
import numpy as np
import evaluate
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

#Environment and Hardware
os.environ["HF_TOKEN"] = "put your key if multiple attemps are required otherwise it can be ignored as well"
torch.set_num_threads(8)
print("Training for Max Accuracy")

MODEL_NAME = "google/mt5-small"
OUTPUT_DIR = "models/translit-pro-final"
MAX_LENGTH = 16

#Setup Tokenizer and Data
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    inputs = [f"transliterate English to {l}: {s}" for l, s in zip(examples['lang'], examples['source'])]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples['target'], max_length=MAX_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Processing all samples")
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

train_dataset = Dataset.from_pandas(train_df).map(preprocess, batched=True, num_proc=4)
val_dataset = Dataset.from_pandas(val_df).map(preprocess, batched=True, num_proc=4)

#to save memory
del train_df
del val_df

#Model and Training Arguments
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=4e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    predict_with_generate=False,
    use_cpu=True,
    optim="adafactor",
    lr_scheduler_type="cosine",
    warmup_steps=500,
    logging_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

if __name__ == "__main__":
    print("Start Training"")
    trainer.train()
    
    # Save the model
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
    print(f"Training is done. Model saved to {OUTPUT_DIR}")
