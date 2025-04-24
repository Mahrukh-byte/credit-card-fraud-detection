import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_split_data(preprocessed_data_path, train_data_path, test_data_path):
    # Load the preprocessed dataset
    data = pd.read_csv(r'C:\Users\ASUS\OneDrive\Credit Card Fraud Detection\processed_data.csv')

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save train and test sets to CSV files
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    print(f"Train data saved to {train_data_path}")
    print(f"Test data saved to {test_data_path}")

# Example usage
preprocessed_data_path = r'C:\Users\ASUS\OneDrive\Credit Card Fraud Detection\preprocessed_data.csv'
train_data_path = r'C:\Users\ASUS\OneDrive\Credit Card Fraud Detection\train_data.csv'
test_data_path = r'C:\Users\ASUS\OneDrive\Credit Card Fraud Detection\test_data.csv'

preprocess_and_split_data(preprocessed_data_path, train_data_path, test_data_path)