import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
import mlflow
import mlflow.sklearn
import joblib
import os

DATA_PATH   = "app/data/Telco-Customer-Churn.csv"
MODELS_PATH = "app/models"
PLOTS_PATH  = "app/plots"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH,  exist_ok=True)

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df = df.drop('customerID', axis=1)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['is_new_customer']    = (df['tenure'] < 6).astype(int)
    df['total_services']     = (
        df['PhoneService'] + df['MultipleLines'] +
        df['OnlineSecurity'] + df['OnlineBackup'] +
        df['DeviceProtection'] + df['TechSupport'] +
        df['StreamingTV'] + df['StreamingMovies']
    )
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODELS_PATH}/scaler.pkl")
    return X_train, X_test, y_train, y_test, feature_names

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    print(f"\nTraining: {name}...")
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1        = report['1']['f1-score']
        precision = report['1']['precision']
        recall    = report['1']['recall']
        mlflow.log_param("model_type", name)
        mlflow.log_metric("roc_auc",   round(auc, 4))
        mlflow.log_metric("f1",        round(f1, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall",    round(recall, 4))
        mlflow.sklearn.log_model(model, name)
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Stayed', 'Churned'])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f'Confusion Matrix — {name}')
        plt.tight_layout()
        plot_path = f"{PLOTS_PATH}/cm_{name.replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(plot_path)
    return {
        'model':     model,
        'auc':       auc,
        'f1':        f1,
        'precision': precision,
        'recall':    recall
    }

def generate_shap(best_model, best_name, X_train, feature_names):
    print(f"\nGenerating SHAP explanations for {best_name}...")
    try:
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_train[:200])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_train[:200],
            feature_names=feature_names,
            show=False,
            plot_size=(10, 7)
        )
        plt.title(f'SHAP Feature Importance — {best_name}')
        plt.tight_layout()
        plt.savefig(f"{PLOTS_PATH}/shap_summary.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  SHAP plot saved to app/plots/shap_summary.png")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

def run_training():
    print("=" * 40)
    print("MODEL TRAINING PIPELINE")
    print("=" * 40)

    mlflow.set_experiment("churn-prediction")

    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features")

    models = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM":                  SVC(probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate(
            name, model, X_train, X_test, y_train, y_test
        )

    print("\n" + "=" * 40)
    print("RESULTS COMPARISON")
    print("=" * 40)
    results_df = pd.DataFrame({
        name: {
            'ROC-AUC':   round(r['auc'], 4),
            'F1':        round(r['f1'], 4),
            'Precision': round(r['precision'], 4),
            'Recall':    round(r['recall'], 4)
        }
        for name, r in results.items()
    }).T.sort_values('ROC-AUC', ascending=False)
    print(results_df.to_string())

    best_name  = results_df.index[0]
    best_model = results[best_name]['model']
    print(f"\nBest model: {best_name}")
    print(f"Best ROC-AUC: {results_df.loc[best_name, 'ROC-AUC']}")

    joblib.dump(best_model, f"{MODELS_PATH}/model.pkl")
    joblib.dump(feature_names, f"{MODELS_PATH}/feature_names.pkl")
    print(f"Model saved to app/models/model.pkl")
    print(f"Feature names saved to app/models/feature_names.pkl")

    generate_shap(best_model, best_name, X_train, feature_names)

    print("Check app/plots/ for confusion matrices and SHAP plot.")
    print("Run: mlflow ui  — to see experiment dashboard in browser")

if __name__ == "__main__":
    run_training()
