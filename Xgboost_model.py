import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os

def evaluate_with_threshold(y_test, y_proba, threshold):
    """
    Evaluate the model using a custom decision threshold.
    """
    print(f"\nApplying custom decision threshold: {threshold}")
    y_pred_custom = (y_proba > threshold).astype(int)
    print("\nClassification Report (Threshold = {:.2f}):\n".format(threshold), classification_report(y_test, y_pred_custom))
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    print("\nConfusion Matrix (Threshold = {:.2f}):\n".format(threshold), cm_custom)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_custom, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})")
    plt.show()

    return y_pred_custom

def train_and_evaluate_xgboost():
    current_dir = os.path.dirname(__file__)
    train_data_path = r'C:\Users\ASUS\OneDrive\Credit Card Fraud Detection\train_data.csv'
    test_data_path = r'C:\Users\ASUS\OneDrive\Credit Card Fraud Detection\test_data.csv'
    model_save_path = os.path.join(current_dir, 'xgboost_model.pkl')

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']

    print("Training the XGBoost Classifier...")
    model = XGBClassifier(
        random_state=42,
        n_estimators=50,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print("Model training completed.")

    print("Evaluating the model with default threshold (0.5)...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report (Default Threshold = 0.5):\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (Default Threshold = 0.5):\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Default Threshold = 0.5)")
    plt.show()

    for threshold in [0.4, 0.3, 0.6]:
        evaluate_with_threshold(y_test, y_proba, threshold)
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color='blue')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig("roc_curve.png")
    plt.show()

    joblib.dump(model, model_save_path)
    print(f"Trained model saved to {model_save_path}")

if __name__ == "__main__":
    train_and_evaluate_xgboost()