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
   ```bash
   git clone [https://github.com/kshitiz-ds/multilingual-indic-transliteration.git](https://github.com/kshitiz-ds/multilingual-indic-transliteration.git)
   cd devnagri
