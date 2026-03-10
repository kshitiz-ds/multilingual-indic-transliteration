#import neceessary libraries
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import matplotlib.pyplot as plt

#Configuration
MODEL_PATH = "models/translit-pro-final/best_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
cer_metric = load("cer")
wer_metric = load("wer")

def transliterate(text, target_lang):
    """Single word prediction"""
    input_text = f"transliterate English to {target_lang}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=16)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_full_report(test_csv="data/val.csv"):
    print("Generate Performance Report")
    df = pd.read_csv(test_csv).head(500) #500 samples taken
    
    preds = []
    for _, row in df.iterrows():
        preds.append(transliterate(row['source'], row['lang']))
    
    df['prediction'] = preds
    
    #Calculate Global metrics
    overall_cer = cer_metric.compute(predictions=df['prediction'], references=df['target'])
    overall_wer = wer_metric.compute(predictions=df['prediction'], references=df['target'])
    accuracy = (df['prediction'] == df['target']).mean()

    print(f"FINAL REPORT")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Character Error Rate (CER): {overall_cer:.4f}")
    print(f"Word Error Rate (WER): {overall_wer:.4f}")

    #Languagewise breakdown
    for lang in df['lang'].unique():
        lang_df = df[df['lang'] == lang]
        lang_acc = (lang_df['prediction'] == lang_df['target']).mean()
        print(f"Script {lang}: {lang_acc:.2%} Accuracy")

    df[['source', 'lang', 'target', 'prediction']].head(10).to_markdown("eval_sample.md")
    print("Sample comparisons saved to eval_sample.md")

if __name__ == "__main__":
    run_full_report()