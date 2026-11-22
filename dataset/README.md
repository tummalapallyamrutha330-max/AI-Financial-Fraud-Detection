# Credit Card Fraud Detection Dataset

## Dataset Source
This project utilizes the **Credit Card Fraud Detection Dataset** originally released by the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB).

- **Official Kaggle URL:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Direct Mirror:** Available via Hugging Face (`David-Egea/Creditcard-fraud-detection`)

## Dataset Characteristics
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (~0.172% extreme class imbalance)
- **Legitimate Transactions:** 284,315 (~99.828%)
- **Input Features (30 numerical features):**
  - `Time`: Elapsed seconds between this transaction and the first transaction in the dataset.
  - `V1` to `V28`: Principal components obtained via PCA (anonymized for user confidentiality).
  - `Amount`: Transaction monetary value.
- **Target Variable:**
  - `Class`: `0` for legitimate transactions, `1` for fraudulent transactions.

## Setup Instructions

### Option 1: Automated Download (Recommended)
Run the dataset download script from the project root:
```bash
python dataset/download_dataset.py
```
This script downloads `creditcard.csv` directly into the `dataset/` directory and validates row counts and fraud distributions.

### Option 2: Manual Download from Kaggle
1. Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Download `creditcardfraud.zip`.
3. Extract the `creditcard.csv` file into this `dataset/` folder:
   `AI_Fraud_Detection/dataset/creditcard.csv`.
