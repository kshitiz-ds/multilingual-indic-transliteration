# Multilingual Indic Transliteration (mT5-Small)

This repository contains an end-to-end pipeline for transliterating English text into Hindi (**hi**), Bengali (**bn**), and Tamil (**ta**) using a fine-tuned **mT5-Small** model.

## 🚀 Features
- **Model:** Google mT5-Small (approx. 300M parameters).
- **Architecture:** Sequence-to-Sequence (Encoder-Decoder) Transformer.
- **Precision:** Optimized for CPU-only environments with high RAM (51GB+).
- **Batching:** Utilizes a batch size of 32 with a Cosine Learning Rate Scheduler for stable convergence.

---

## 🛠️ Installation

1. **Clone the repository:**
bash git clone [https://github.com/kshitiz-ds/multilingual-indic-transliteration.git](https://github.com/kshitiz-ds/multilingual-indic-transliteration.git)
   cd devnagri

## To install all model dependencies
pip install -r requirements.txt

## To prepare the multilingual training set (50k samples per language)
python3 data_prep.py

## To start the fine-tuning process (Estimated duration: ~24-29 hours on 8-core CPU)
nohup python3 train.py > training.log 2>&1 &

## To test the build model
#Once training is complete, the best model is saved to models/translit-pro-final/best_model
python3 predict.py
