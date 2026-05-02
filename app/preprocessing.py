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
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # New customers have 0 tenure and empty TotalCharges — fill with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print(f"  TotalCharges nulls remaining: {df['TotalCharges'].isnull().sum()}")
    
    print("Step 2 — Dropping customerID...")
    df = df.drop('customerID', axis=1)
    
    print("Step 3 — Encoding target column Churn...")
   
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    print(f"  Churn value counts: {df['Churn'].value_counts().to_dict()}")
    
    return df

def encode_categories(df):
    print("Step 4 — Encoding categorical columns...")
    
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
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    print("  Created: charges_per_tenure")
    
    # Feature 2: is this a new customer?
    # EDA showed new customers churn much more

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
    
    X = df.drop('Churn', axis=1)  
    y = df['Churn']    
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  
    X_test  = scaler.transform(X_test)  
 
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
