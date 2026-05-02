import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH  = "app/data/Telco-Customer-Churn.csv"
PLOTS_PATH = "app/plots"
os.makedirs(PLOTS_PATH, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
    return df

def plot_churn_by_contract(df):
    plt.figure(figsize=(8, 5))
    churn_by_contract = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    churn_by_contract.columns = ['Contract', 'Churn Rate (%)']
    sns.barplot(data=churn_by_contract, x='Contract', y='Churn Rate (%)', palette='Blues_d')
    plt.title('Churn Rate by Contract Type')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/churn_by_contract.png", dpi=150)
    plt.close()
    print("Saved: churn_by_contract.png")

def plot_churn_by_tenure(df):
    plt.figure(figsize=(8, 5))
    churned     = df[df['Churn'] == 'Yes']['tenure']
    not_churned = df[df['Churn'] == 'No']['tenure']
    plt.hist(not_churned, bins=30, alpha=0.6, label='Stayed',  color='steelblue')
    plt.hist(churned,     bins=30, alpha=0.6, label='Churned', color='tomato')
    plt.title('Customer Tenure vs Churn')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/churn_by_tenure.png", dpi=150)
    plt.close()
    print("Saved: churn_by_tenure.png")

def plot_monthly_charges(df):
    plt.figure(figsize=(8, 5))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges',
                palette={'No': 'steelblue', 'Yes': 'tomato'})
    plt.title('Monthly Charges: Churned vs Retained')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/monthly_charges.png", dpi=150)
    plt.close()
    print("Saved: monthly_charges.png")

def run_eda():
    print("=" * 40)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 40)
    df = load_data()
    print("Generating plots...")
    plot_churn_by_contract(df)
    plot_churn_by_tenure(df)
    plot_monthly_charges(df)
    print("Check app/plots/ for your 3 charts.")
    return df

if __name__ == "__main__":
    run_eda()
