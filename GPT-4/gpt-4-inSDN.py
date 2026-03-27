"""
InSDN Research-Grade IDS Pipeline
Author: SDN ML Cybersecurity Research

Includes:
- Binary & Multiclass classification
- Stratified K-Fold Cross Validation
- Optuna hyperparameter tuning
- ROC-AUC curves
- SHAP explainability
- Model persistence
- SDN controller inference pipeline
"""
from pathlib import Path
# =========================
# FILE PATHS (EDIT HERE)
# =========================

PROJECT_ROOT = Path(__file__).resolve()


while not (PROJECT_ROOT / "Deepseek-V3").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

normal_path = PROJECT_ROOT / 'Deepseek-V3' / 'inSDN' / 'Normal_data.csv'
ovs_path = PROJECT_ROOT / 'Deepseek-V3' / 'inSDN' / 'OVS.csv'
metasploitable_path = PROJECT_ROOT / 'Deepseek-V3' / 'inSDN' / 'metasploitable-2.csv'

nrows = None
# =========================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import shap
import optuna
import joblib


# =========================
# DATA LOADING
# =========================

def detect_label_column(df):
    keywords = ["label", "class", "attack", "category", "type"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    raise ValueError("Label column not found")

def load_and_merge():
    normal = pd.read_csv(normal_path, nrows=nrows)
    ovs = pd.read_csv(ovs_path, nrows=nrows)
    meta = pd.read_csv(metasploitable_path, nrows=nrows)

    label_col = detect_label_column(ovs)

    normal[label_col] = "Benign"

    common_cols = list(set(normal.columns) &
                       set(ovs.columns) &
                       set(meta.columns))

    df = pd.concat([
        normal[common_cols],
        ovs[common_cols],
        meta[common_cols]
    ], ignore_index=True)

    return df, label_col


# =========================
# CLEANING
# =========================

def clean_data(df):

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    nunique = df.nunique()
    df.drop(columns=nunique[nunique <= 1].index, inplace=True)

    drop_patterns = ["ip", "flow id", "timestamp", "time"]
    drop_cols = [c for c in df.columns if any(p in c.lower() for p in drop_patterns)]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    df.fillna(0, inplace=True)

    return df


# =========================
# OPTUNA TUNING
# =========================

def tune_xgb(X, y):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": 42
        }

        model = XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X, y,
                                cv=cv,
                                scoring="f1_weighted",
                                n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best XGBoost params:", study.best_params)
    return study.best_params


# =========================
# ROC CURVES
# =========================

def plot_binary_roc(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"Binary ROC Curve (AUC={auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    save_path = RESULTS_DIR / "binary_roc_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

def save_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                xticklabels=labels,
                yticklabels=labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    save_path = RESULTS_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def plot_multiclass_roc(model, X_test, y_test, n_classes):
    probs = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, probs, multi_class="ovr")

    print("Multiclass ROC-AUC (OVR):", auc)


# =========================
# SHAP ANALYSIS
# =========================

def shap_analysis(model, X_sample):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Bar plot
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    save_path_bar = RESULTS_DIR / "shap_bar.png"
    plt.savefig(save_path_bar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_bar}")

    # Full summary
    shap.summary_plot(shap_values, X_sample, show=False)
    save_path_summary = RESULTS_DIR / "shap_summary.png"
    plt.savefig(save_path_summary, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_summary}")


# =========================
# INFERENCE PIPELINE
# =========================

class SDNInferencePipeline:

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def preprocess(self, df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def predict(self, flow_features_df):
        flow_features_df = self.preprocess(flow_features_df)
        return self.model.predict(flow_features_df)


# =========================
# MAIN
# =========================

def main():

    df, label_col = load_and_merge()
    df = clean_data(df)

    df["Label_Multi"] = df[label_col]
    df["Label_Binary"] = df["Label_Multi"].apply(
        lambda x: "Benign" if x.lower() == "benign" else "Attack"
    )

    df.drop(columns=[label_col], inplace=True)

    X = df.drop(columns=["Label_Binary", "Label_Multi"])

    le_bin = LabelEncoder()
    y_bin = le_bin.fit_transform(df["Label_Binary"])

    le_multi = LabelEncoder()
    y_multi = le_multi.fit_transform(df["Label_Multi"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, stratify=y_bin, test_size=0.2, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print("Tuning XGBoost...")
    best_params = tune_xgb(X_train, y_train)

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    plot_binary_roc(model, X_test, y_test)

    shap_analysis(model, X_train.sample(1000))

    joblib.dump(model, "best_insdn_model.pkl")
    print("Model saved as best_insdn_model.pkl")

    save_confusion_matrix(
        y_test,
        preds,
        le_bin.classes_,
        "binary_confusion_matrix.png"
    )

    # Example SDN inference usage:
    # pipeline = SDNInferencePipeline("best_insdn_model.pkl")
    # pipeline.predict(new_flow_dataframe)


if __name__ == "__main__":
    main()