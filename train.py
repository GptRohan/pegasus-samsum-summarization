# ============================================================
# PEGASUS Summarization Fine-tuning on SAMSum Dataset
# ============================================================

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)

# ------------------------------------------------------------
# 1. Device setup
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------------------------------------
# 2. Load model & tokenizer
# ------------------------------------------------------------
model_name = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# ------------------------------------------------------------
# 3. Load SAMSum dataset
# ------------------------------------------------------------
dataset = load_dataset("knkarthick/samsum")

# ------------------------------------------------------------
# 4. Tokenization function
# ------------------------------------------------------------
def convert_ex_to_features(batch):
    # Tokenize dialogue
    inputs = tokenizer(
        batch["dialogue"],
        max_length=512,
        truncation=True
    )
    
    # Tokenize summary
    targets = tokenizer(
        text_target=batch["summary"],
        max_length=128,
        truncation=True
    )
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }

# Map preprocessing
tokenized_data = dataset.map(convert_ex_to_features, batched=True)

# ------------------------------------------------------------
# 5. Data collator
# ------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# ------------------------------------------------------------
# 6. TrainingArguments
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="pegasus-samsum",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    logging_steps=50,
    save_steps=5000,
    evaluation_strategy="steps",
    eval_steps=500,
    report_to="none"
)

# ------------------------------------------------------------
# 7. Trainer Setup
# ------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"]
)

# ------------------------------------------------------------
# 8. Train
# ------------------------------------------------------------
trainer.train()

print("Training Done!")
