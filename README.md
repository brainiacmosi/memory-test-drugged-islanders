# Memory Test Analysis on Drugged Islanders

## Project Overview
A comprehensive data analysis and machine learning project investigating the effects of different drugs (Alprazolam, Triazolam, Placebo) on memory performance. This project demonstrates the full data science pipeline from exploratory analysis to predictive modeling.

## Key Findings
- **Alprazolam** shows strong dose-dependent cognitive enhancement (+22.6 points at high dose)
- **Triazolam** demonstrates consistent negative effects on memory
- **Machine Learning Model** achieved 77.2% AUC in predicting treatment response
- **100% improvement rate** with high-dose Alprazolam

## Technical Skills Demonstrated
- **Data Cleaning & Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Statistical Analysis** (ANOVA, t-tests, post-hoc tests)
- **Data Visualization** (Matplotlib, Seaborn)
- **Machine Learning** (Logistic Regression, Random Forest, XGBoost)
- **Model Evaluation** (ROC-AUC, Confusion Matrix, Feature Importance)

## Results Summary
| Metric | Value |
|--------|-------|
| Best Model | Logistic Regression |
| ROC-AUC | 0.772 |
| Accuracy | 70.0% |
| Recall | 91.3% |
| Precision | 67.7% |

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/Memory-Test-Drug-Analysis.git

# Install dependencies
pip install -r requirements.txt

# Run analysis
python notebooks/memory_test_analysis_v3.py
