import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH   = "app/data/Telco-Customer-Churn.csv"
MODELS_PATH = "app/models"
os.makedirs(MODELS_PATH, exist_ok=True)

def load_and_clean(df):
    print("Step 1 — Fixing TotalCharges...")
    # TotalCharges is stored as text — convert to number
    # errors='coerce' means: if it cant convert, put NaN instead of crashing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # New customers have 0 tenure and empty TotalCharges — fill with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print(f"  TotalCharges nulls remaining: {df['TotalCharges'].isnull().sum()}")
    
    print("Step 2 — Dropping customerID...")
    # customerID is just a unique ID — useless for prediction
    df = df.drop('customerID', axis=1)
    
    print("Step 3 — Encoding target column Churn...")
    # Convert Yes/No to 1/0
    # This is our target — what we want to predict
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    print(f"  Churn value counts: {df['Churn'].value_counts().to_dict()}")
    
    return df

def encode_categories(df):
    print("Step 4 — Encoding categorical columns...")
    # LabelEncoder converts text categories to numbers
    # Example: Month-to-month=0, One year=1, Two year=2
    
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"  Columns to encode: {cat_cols}")
    
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        print(f"  Encoded: {col}")
    
    return df

def engineer_features(df):
    print("Step 5 — Engineering new features...")
    
    # Feature 1: charges per month of tenure
    # A customer paying $100/month for 1 month is different from
    # one paying $100/month for 5 years — this ratio captures that
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    print("  Created: charges_per_tenure")
    
    # Feature 2: is this a new customer?
    # Our EDA showed new customers churn much more
    # Customers with less than 6 months tenure = new customer
    df['is_new_customer'] = (df['tenure'] < 6).astype(int)
    print("  Created: is_new_customer")
    
    # Feature 3: has multiple services?
    # Customers using more services are more invested — less likely to churn
    df['total_services'] = (
        df['PhoneService'] + df['MultipleLines'] +
        df['OnlineSecurity'] + df['OnlineBackup'] +
        df['DeviceProtection'] + df['TechSupport'] +
        df['StreamingTV'] + df['StreamingMovies']
    )
    print("  Created: total_services")
    
    return df

def split_and_scale(df):
    print("Step 6 — Splitting into train and test sets...")
    
    X = df.drop('Churn', axis=1)  # everything except target
    y = df['Churn']               # just the target
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # stratify=y means keep same churn ratio in both train and test
    # random_state=42 means results are reproducible every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"  Training set: {X_train.shape[0]} rows")
    print(f"  Test set:     {X_test.shape[0]} rows")
    print(f"  Train churn rate: {y_train.mean():.1%}")
    print(f"  Test churn rate:  {y_test.mean():.1%}")
    
    print("Step 7 — Scaling features...")
    # StandardScaler makes all numbers have mean=0 and std=1
    # Without this, tenure (0-72) dominates MonthlyCharges (18-118)
    # The model would think tenure is more important just because its bigger
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # learn scale from train only
    X_test  = scaler.transform(X_test)       # apply same scale to test
    
    # Save scaler — we need this exact same scaler in Streamlit later
    joblib.dump(scaler, 'app/models/scaler.pkl')
    print("  Scaler saved to app/models/scaler.pkl")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def run_preprocessing():
    print("=" * 40)
    print("PREPROCESSING PIPELINE")
    print("=" * 40)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    df = load_and_clean(df)
    df = encode_categories(df)
    df = engineer_features(df)
    
    X_train, X_test, y_train, y_test, feature_names = split_and_scale(df)
    
    print("\nFeature names going into model:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")
    
    print(" Data is ready for model training.")
    return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    run_preprocessing()
