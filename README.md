# PEGASUS SAMSum Summarization

This project fine-tunes **Google PEGASUS (pegasus-xsum)** on the **SAMSum** dataset to generate summaries of multi-person chat conversations.

## 🚀 Features
- Sequence-to-sequence summarization
- HuggingFace Trainer API
- PEGASUS model
- SAMSum dataset (dialogue → summary)
- CUDA/GPU support

## 📁 Files
- `train.py` → Full training script
- `requirements.txt` → Project dependencies

## 📦 Installation
pip install -r requirements.txt

## ▶️ Run Training
python train.py

## 📝 Dataset
Dataset used: **SAMSum**  
Contains chat conversations and human-written summaries.

## 🎯 Goal
Given a full dialogue → model returns a short summary.
# pegasus-samsum-summarization
