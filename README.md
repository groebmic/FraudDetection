# 💳 Credit Card Fraud Detection with Holiday Feature

## Project Description

This project predicts whether a credit card transaction is fraudulent using key features like transaction amount (`amt`) and whether the transaction occurred on a public holiday (`is_holiday`). It combines multiple datasets across different years and applies machine learning models to handle class imbalance and extract predictive insights.

## Results

The Logistic Regression model achieved higher recall for detecting fraudulent transactions compared to Random Forest, though with lower precision. Due to the heavy class imbalance (fraud cases <1%), maximizing recall was prioritized to reduce false negatives.

---

## Name & URL

| Name         | URL                         |
|--------------|-----------------------------|
| Gradio App   | [Hugging Face Space](https://huggingface.co/spaces/groebmic/FraudPrediction) |
| Code         | [GitHub Repository](https://github.com/groebmic/FraudDetection/)           |

---

## Data Sources and Features Used Per Source

| Data Source           | Features                             |
|-----------------------|--------------------------------------|
| creditcard.csv        | `Amount`, `Time`, `Class` (0/1)      |
| fraudTrain.csv        | `amt`, `trans_date_trans_time`, `is_fraud` |
| fraudTest.csv         | `amt`, `trans_date_trans_time`, `is_fraud` |
| eu_holidays.csv       | `Date`, `is_holiday`                 |

---

## Features Created

| Feature      | Description                                                  |
|--------------|--------------------------------------------------------------|
| `amt`        | Transaction amount                                           |
| `is_holiday` | 1 if transaction occurred on an official EU holiday, else 0 |

---

## Model Training

### Dataset Sizes
- Combined transactions: ~1 million entries
- Fraud cases: ~0.5–1% of total

### Data Splitting
- Train/Test: 80/20 split

### Models
- **Random Forest**
  - `class_weight='balanced'`
  - Good at handling non-linear feature interactions
  - 
- **Logistic Regression**
  - `class_weight='balanced'`
  - Higher recall for fraud class

---

## Performance

| Model             | Recall (1) | Precision (1) | F1-Score (1) | Accuracy |
|-------------------|------------|---------------|--------------|----------|
| Logistic Regression | 0.75       | 0.06          | 0.11         | 0.95     |
| Random Forest       | 0.31       | 0.03          | 0.05         | 0.94     |

---

## Demo App UI (Gradio)

The Gradio web interface (`app.py`) allows users to:

- Select the model (Random Forest or Logistic Regression)
- Input:
  - `amt` (Transaction amount)
  - `is_holiday` (0 or 1)
- Output:
  - Fraud prediction (yes/no)
  - Confidence probability

To run locally:

```bash
python app.py
