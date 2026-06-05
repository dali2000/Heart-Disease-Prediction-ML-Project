import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import joblib
import os

EXPERIMENT_NAME = "HeartDisease_Optimized"

def prepare_data(file_path="data/heart_disease.csv"):
    """Load and split the Heart Disease dataset."""
    df = pd.read_csv(file_path)

    # All columns are numeric — no categorical encoding needed
    X = df.drop('target', axis=1)
    y = df['target']

    numerical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ])

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    filename = f"data/confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    filename = f"data/roc_curve_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def run_optimized_experiment(model_name, model, param_grid):
    (X_train, X_test, y_train, y_test), preprocessor = prepare_data()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        # Stacking ensemble
        if model == "STACKING_PLACEHOLDER":
            base_learners = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, max_depth=3,
                                      eval_metric='logloss', random_state=42))
            ]
            model = StackingClassifier(
                estimators=base_learners,
                final_estimator=LogisticRegression(),
                cv=5
            )

        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                                   scoring='f1', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        mlflow.log_params(best_params)

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print(f"[{model_name}] Best Params: {best_params}")
        print(f"[{model_name}] Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=['No Disease', 'Disease']))

        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)

        roc_path = plot_roc_curve(y_test, y_prob, model_name)
        mlflow.log_artifact(roc_path)

        mlflow.sklearn.log_model(best_model, "model")

        return best_model, f1

if __name__ == "__main__":
    models_config = {
        "RandomForest_SMOTE": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        "XGBoost_SMOTE": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6]
            }
        },
        "Stacking_Ensemble": {
            "model": "STACKING_PLACEHOLDER",
            "params": {
                'classifier__cv': [5]
            }
        }
    }

    best_overall_model = None
    best_overall_f1 = -1

    for name, config in models_config.items():
        print(f"\nRunning {name}...")
        model, f1 = run_optimized_experiment(name, config["model"], config["params"])
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = model

    if best_overall_model:
        joblib.dump(best_overall_model, "data/best_model_pipeline.pkl")
        print(f"\nBest model saved to data/best_model_pipeline.pkl | F1={best_overall_f1:.4f}")
