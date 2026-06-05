import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_advanced_eda(file_path="data/heart_disease.csv"):
    """
    Performs Advanced Exploratory Data Analysis on the Heart Disease dataset.
    Dataset: UCI Heart Disease (Cleveland) - 303 patients, 14 features.
    Target: 0 = No Disease, 1 = Disease
    """
    df = pd.read_csv(file_path)

    print("--- Dataset Shape ---")
    print(df.shape)
    print("\n--- First rows ---")
    print(df.head())
    print("\n--- Info ---")
    print(df.info())
    print("\n--- Describe ---")
    print(df.describe())
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # 1. Target Distribution
    plt.figure(figsize=(8, 5))
    df['target'].value_counts(normalize=True).plot(
        kind='bar', color=['skyblue', 'salmon'], edgecolor='white'
    )
    plt.title('Heart Disease Proportion (0: No Disease, 1: Disease)')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.savefig('data/target_imbalance.png', bbox_inches='tight')
    plt.close()

    # 2. Categorical Impact on Target
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    axes = axes.flatten()

    for i, feature in enumerate(categorical_features):
        sns.countplot(x=feature, hue='target', data=df, ax=axes[i], palette='muted')
        axes[i].set_title(f'Heart Disease by {feature}')
        axes[i].legend(title='Disease', labels=['No', 'Yes'], loc='upper right')

    plt.tight_layout()
    plt.savefig('data/categorical_impact.png', bbox_inches='tight')
    plt.close()

    # 3. Numerical Boxplots
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(y=df[feature], x=df['target'], hue=df['target'],
                    palette='Set2', legend=False)
        plt.title(f'{feature} Distribution by Disease')
        plt.xlabel('Heart Disease (0=No, 1=Yes)')

    plt.tight_layout()
    plt.savefig('data/numerical_boxplots.png', bbox_inches='tight')
    plt.close()

    # 4. Pairplot (sampled)
    sample_cols = ['age', 'thalach', 'oldpeak', 'chol', 'trestbps', 'target']
    sns.pairplot(df[sample_cols].sample(min(200, len(df))),
                 hue='target', diag_kind='kde', palette='husl')
    plt.savefig('data/multivariate_pairplot.png', bbox_inches='tight')
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('data/correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    # 6. Age distribution by disease
    plt.figure(figsize=(10, 5))
    df[df['target'] == 0]['age'].plot(kind='hist', alpha=0.6, bins=20,
                                       label='No Disease', color='skyblue')
    df[df['target'] == 1]['age'].plot(kind='hist', alpha=0.6, bins=20,
                                       label='Disease', color='salmon')
    plt.title('Age Distribution by Heart Disease Status')
    plt.xlabel('Age')
    plt.legend()
    plt.savefig('data/age_distribution.png', bbox_inches='tight')
    plt.close()

    print("\nAdvanced EDA completed. Plots saved in 'data/' directory.")

if __name__ == "__main__":
    run_advanced_eda()
