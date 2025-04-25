# credit-card-fraud-detection
 A Machine learning project on Credit Card Fraud Detection
# Credit Card Fraud Detection using XGBoost

This project implements a machine learning pipeline to detect fraudulent credit card transactions using the XGBoost classifier. The dataset is highly imbalanced, with very few fraud cases, and the project tackles this challenge using appropriate model tuning and threshold analysis.

---

## Dataset

The dataset is divided into two parts:
- `train_data.csv`: Used to train the model.
- `test_data.csv`: Used for evaluation.

Each transaction is represented by numerical features (likely PCA-transformed), with a target column named `Class`:
- `0` = Legitimate
- `1` = Fraudulent

---

## Preprocessing

- No null value handling required (data is already clean)
- Features and target (`Class`) are separated
- `scale_pos_weight` parameter is used in XGBoost to handle class imbalance

---

## Model: XGBoost Classifier

Key parameters:
- `n_estimators=50`
- `max_depth=5`
- `learning_rate=0.05`
- `scale_pos_weight = ratio of negative to positive samples`
- `eval_metric='logloss'`

The model is trained on the training dataset and evaluated on the test set.

---

## Evaluation

### 1. **Default Threshold = 0.5**
- Classification Report and Confusion Matrix are generated.
- Model shows performance metrics like precision, recall, F1-score.

### 2. **Custom Thresholds (0.4, 0.3, 0.6)**
- To analyze sensitivity to fraud detection, custom thresholds are applied.
- Confusion matrices are visualized for each threshold.

### 3. **ROC Curve & AUC Score**
- ROC Curve is plotted.
- AUC (Area Under Curve) is calculated and displayed.
- 
## Model Saving

- The trained model is saved using `joblib` as:  
  `models/xgboost_model.pkl`

## Output Plot

- ROC Curve is saved as `plots/roc_curve.png`

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
