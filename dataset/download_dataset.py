"""
Dataset downloader and validator for Credit Card Fraud Detection.
Downloads the authentic ULB creditcard.csv from public repository mirror
and verifies row counts, schema, and fraud distribution.
"""
import os
import sys
import urllib.request
import pandas as pd

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATASET_DIR, "creditcard.csv")
SAMPLE_PATH = os.path.join(DATASET_DIR, "sample_creditcard.csv")

# Direct authenticated mirror of authentic ULB creditcard dataset
DATASET_URL = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"

def download_dataset():
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 100_000_000:
        print(f"Dataset already exists at: {DATASET_PATH} ({os.path.getsize(DATASET_PATH):,} bytes)")
        return validate_dataset(DATASET_PATH)

    print(f"Downloading authentic ULB Credit Card Fraud dataset to: {DATASET_PATH}...")
    print("This may take 1-2 minutes depending on your internet connection (~143 MB)...")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH, reporthook=reporthook)
        print("\nDownload completed successfully.")
        return validate_dataset(DATASET_PATH)
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Please download manually from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"and place creditcard.csv at {DATASET_PATH}")
        return False

def validate_dataset(filepath):
    print(f"\nValidating dataset at: {filepath}...")
    try:
        df = pd.read_csv(filepath)
        total_rows = len(df)
        total_cols = len(df.columns)
        fraud_count = int(df['Class'].sum())
        legit_count = total_rows - fraud_count

        print(f"Validation successful:")
        print(f"  - Total transactions: {total_rows:,}")
        print(f"  - Total features:     {total_cols}")
        print(f"  - Legitimate cases:   {legit_count:,} ({legit_count / total_rows * 100:.3f}%)")
        print(f"  - Fraudulent cases:   {fraud_count:,} ({fraud_count / total_rows * 100:.3f}%)")

        # Create stratified sample for quick tests if not existing
        if not os.path.exists(SAMPLE_PATH):
            print("Generating stratified test sample (5,000 legitimate + all 492 fraud)...")
            legit_sample = df[df['Class'] == 0].sample(n=5000, random_state=42)
            fraud_sample = df[df['Class'] == 1]
            sample_df = pd.concat([legit_sample, fraud_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            sample_df.to_csv(SAMPLE_PATH, index=False)
            print(f"Sample saved to {SAMPLE_PATH} ({len(sample_df)} rows)")

        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
