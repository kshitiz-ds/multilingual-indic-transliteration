#import necessary libraries
import json
import os
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sklearn.model_selection import train_test_split

LANGUAGES = ['hin', 'ben', 'tam']
NORM_MAP = {'hin': 'hi', 'ben': 'bn', 'tam': 'ta'}
SAMPLE_SIZE = 50_000  # Number of rows per language
OUTPUT_DIR = 'data'
RANDOM_SEED = 44

def parse_json_bytes(raw: bytes) -> list:
    text = raw.decode('utf-8', errors='replace').strip()
    records = []
    # Standard JSONL parsing
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        try: records.append(json.loads(line))
        except: pass
    return records

def load_and_sample_language(lang: str, normalizer) -> pd.DataFrame:
    print(f"\n📦  Processing: {lang}")
    zip_path = hf_hub_download(
        repo_id="ai4bharat/Aksharantar",
        filename=f"{lang}.zip",
        repo_type="dataset",
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        train_file = next(f for f in zf.namelist() if 'train' in f.lower())
        raw_bytes = zf.read(train_file)

    records = parse_json_bytes(raw_bytes)
    df = pd.DataFrame(records)
    
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
        print(f"Sampled {SAMPLE_SIZE:,} rows from {len(records):,}")

    src_col = 'english word' if 'english word' in df.columns else df.columns[2]
    tgt_col = 'native word' if 'native word' in df.columns else df.columns[1]

    df['source'] = df[src_col].astype(str).str.strip().str.lower()
    df['target'] = df[tgt_col].apply(lambda x: normalizer.normalize(str(x).strip()))
    df['lang'] = NORM_MAP[lang]

    return df[df['source'].str.len().gt(0) & df['target'].str.len().gt(0)][['source', 'target', 'lang']]

def prepare_medium_data():
    print("Starting Data Prep")
    factory = IndicNormalizerFactory()
    all_data = []

    for lang in LANGUAGES:
        try:
            norm = factory.get_normalizer(NORM_MAP[lang])
            df_clean = load_and_sample_language(lang, norm)
            all_data.append(df_clean)
            print(f"Cleaned {len(df_clean):,} pairs")
        except Exception as e:
            print(f"Error processing {lang}: {e}")

    full_df = pd.concat(all_data, ignore_index=True)
    
    #Stratified split
    train_df, temp_df = train_test_split(
        full_df, test_size=0.10, random_state=RANDOM_SEED, stratify=full_df['lang']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df['lang']
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
    
    print(f"Total Rows Saved: {len(full_df):,}")
    print(f"Data saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    prepare_medium_data()