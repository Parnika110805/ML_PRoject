import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)



DEFAULT_PATHS = {
    "toddler":     "data/Autism Dataset for Toddlers.csv",
    "early_child": "data/Autism-Child-Data.csv",
    "adolescent":  "data/adolescent_ASD.csv",
}

MODEL_OUT_DIR = "model"


RENAME_TODDLER = {
    "Family_mem_with_ASD":    "Family_ASD",
    "Who completed the test": "Completed_by",
    "Class/ASD Traits":       "ASD",
    "Class/ASD_Traits":       "ASD",
    "Class/ASD_Traits ":      "ASD",
    "Case_No":                "case_no",
}

RENAME_EARLY_CHILD = {
    "A1_Score":  "A1",  "A2_Score":  "A2",  "A3_Score":  "A3",
    "A4_Score":  "A4",  "A5_Score":  "A5",  "A6_Score":  "A6",
    "A7_Score":  "A7",  "A8_Score":  "A8",  "A9_Score":  "A9",
    "A10_Score": "A10",
    "id":        "case_no",
    "A10_Score": "A10",
    "age":       "Age_Years",
    "gender":    "Sex",
    "ethnicity": "Ethnicity",
    "jundice":   "Jaundice",
    "contry_of_res":   "Country",
    "used_app_before": "Used_App",
    "age_desc":        "Age_Desc",
    "Family_mem_with_ASD":   "Family_ASD",
    "relation":              "Completed_by",
    "Class/ASD":             "ASD",
}

RENAME_ADOLESCENT = {
    "A1_Score":  "A1",  "A2_Score":  "A2",  "A3_Score":  "A3",
    "A4_Score":  "A4",  "A5_Score":  "A5",  "A6_Score":  "A6",
    "A7_Score":  "A7",  "A8_Score":  "A8",  "A9_Score":  "A9",
    "A10_Score": "A10",
    "age":       "Age_Years",
    "gender":    "Sex",
    "ethnicity": "Ethnicity",
    "jundice":   "Jaundice",
    "austim":    "Family_ASD",
    "relation":  "Completed_by",
    "Class/ASD": "ASD",
    "id":              "case_no",
    "contry_of_res":   "Country",
    "used_app_before": "Used_App",
    "age_desc":        "Age_Desc",
    "ASD_traits":      "ASD",
}

DROP_COLS = [
    "case_no",
    "Country",
    "Used_App",
    "Age_Desc",
    "source_group",
]



YES_NO_MAP = {
    "yes": 1, "no": 0,
    "1":   1, "0":  0,
    "1.0": 1, "0.0": 0,
    "true": 1, "false": 0,
    "y":   1, "n":   0,
}

def encode_binary(series: pd.Series, default: int = 0) -> pd.Series:
    return (
        series.astype(str).str.lower().str.strip()
        .map(YES_NO_MAP)
        .fillna(default)
        .astype(int)
    )

def find_target_column(df):
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["asd", "autism", "class"])
        and "a10" not in c.lower()
    ]
    return candidates[0] if candidates else None

def save_plot(name):
    path = os.path.join("plots", f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved plot → {path}")

def load_single_csv(csv_path: str, group_label: str) -> pd.DataFrame:
    print(f"  ▸ Loading [{group_label}]  →  {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"    ✗ File not found: {csv_path} — skipping.")
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    rename_map = {
        "toddler":     RENAME_TODDLER,
        "early_child": RENAME_EARLY_CHILD,
        "adolescent":  RENAME_ADOLESCENT,
    }[group_label]
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["source_group"] = group_label

    for i in range(1, 11):
        col = f"A{i}"
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0).clip(0, 1).astype(int)
            )
        else:
            df[col] = 0



    if group_label == "toddler" and "Age_Mons" in df.columns:
        df["Age_Years"] = pd.to_numeric(df["Age_Mons"], errors="coerce") / 12
    elif "Age_Years" in df.columns:
        df["Age_Years"] = pd.to_numeric(df["Age_Years"], errors="coerce")
    else:
        df["Age_Years"] = np.nan

    default_age = {"toddler": 2.0, "early_child": 2.5, "adolescent": 13.0}[group_label]
    median_age  = df["Age_Years"].median()
    df["Age_Years"] = df["Age_Years"].fillna(
        median_age if not np.isnan(median_age) else default_age
    )


    sex_col = next((c for c in ["Sex", "sex"] if c in df.columns), None)
    if sex_col:
        sex_map = {"m": 1, "male": 1, "f": 0, "female": 0, "1": 1, "0": 0}
        df["Sex_M"] = (
            df[sex_col].astype(str).str.lower().str.strip()
            .map(sex_map).fillna(0).astype(int)
        )
    else:
        df["Sex_M"] = 0


    eth_col = next((c for c in ["Ethnicity", "ethnicity"] if c in df.columns), None)
    if eth_col:
        eth = df[eth_col].astype(str).str.lower().str.strip()
        df["Ethnicity_WE"] = eth.isin(
            ["white european", "white-european", "white_european"]
        ).astype(int)
        df["Ethnicity_SA"] = eth.isin(
            ["south asian", "south-asian", "south_asian", "asian"]
        ).astype(int)
        df["Ethnicity_ME"] = eth.isin(
            ["middle eastern", "middle-eastern", "middle_eastern"]
        ).astype(int)
    else:
        df["Ethnicity_WE"] = df["Ethnicity_SA"] = df["Ethnicity_ME"] = 0


    for col in ["Jaundice", "Family_ASD"]:
        if col in df.columns:
            df[col] = encode_binary(df[col], default=0)
        else:
            df[col] = 0


    clinical_cols = {
        "Speech_Delay":      0,
        "Learning_Disorder": 0,
        "Genetic":           0,
        "Depression":        0,
        "Global_Dev_Delay":  0,
        "Social_Issues":     0,
        "Anxiety":           0,
    }
    for col, default in clinical_cols.items():
        if col in df.columns:
            df[col] = encode_binary(df[col], default)
        else:
            df[col] = default



    if "Completed_by" in df.columns:
        df["Completed_HCP"] = (
            df["Completed_by"].astype(str).str.lower()
            .str.contains(
                r"health|hcp|professional|doctor|clinician",
                regex=True, na=False
            ).astype(int)
        )
    else:
        df["Completed_HCP"] = 0


    if "ASD" not in df.columns:
        target_col = find_target_column(df)
        if target_col:
            df.rename(columns={target_col: "ASD"}, inplace=True)
            print(f"    ⚠ Auto-mapped target: '{target_col}' → ASD")
        else:
            raise ValueError(
                f"[{group_label}] Target column not found after renaming.\n"
                f"  Columns present: {sorted(df.columns.tolist())}"
            )

    if "ASD" not in df.columns:
        raise ValueError(
            f"[{group_label}] Target column not found after renaming.\n"
            f"  Columns present: {sorted(df.columns.tolist())}"
        )

    target_map = {
        "yes": 1, "no": 0,
        "1":   1, "0":  0,
        "1.0": 1, "0.0": 0,
        "true": 1, "false": 0,
    }
    df["ASD"] = df["ASD"].astype(str).str.lower().str.strip().map(target_map)
    n_before = len(df)
    df = df.dropna(subset=["ASD"])
    if len(df) < n_before:
        print(f"    ⚠  Dropped {n_before - len(df)} rows with unreadable ASD label.")
    df["ASD"] = df["ASD"].astype(int)


    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    print(
        f"    ✓ {len(df):>5} rows  |  "
        f"ASD+ {df['ASD'].sum():>4}  |  "
        f"ASD- {(df['ASD']==0).sum():>4}"
    )
    return df


def load_and_merge(paths: dict) -> pd.DataFrame:
    print("\n" + "═" * 64)
    print("  LOADING DATASETS")
    print("═" * 64)

    frames = []
    for label, path in paths.items():
        if path is None:
            continue
        df = load_single_csv(path, label)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No datasets loaded. Check your file paths.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  ► Merged total  : {len(combined)} samples")
    print(f"  ► ASD positive  : {combined['ASD'].sum()}"
          f"  ({combined['ASD'].mean()*100:.1f} %)")
    print(f"  ► ASD negative  : {(combined['ASD']==0).sum()}"
          f"  ({(1-combined['ASD'].mean())*100:.1f} %)")
    return combined



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["A_comm"]        = df["A1"] + df["A2"] + df["A9"]
    df["A_social"]      = df["A3"] + df["A4"] + df["A7"]
    df["A_imaginative"] = df["A5"] + df["A6"]
    df["A_sensory"]     = df["A8"]

    df["A10_blank"]     = df["A10"]


    df["Speech_x_Learning"] = df["Speech_Delay"]   * df["Learning_Disorder"]
    df["Genetic_x_GDD"]     = df["Genetic"]         * df["Global_Dev_Delay"]
    df["Anxiety_x_Dep"]     = df["Anxiety"]          * df["Depression"]

    df["Qchat_x_Social"] = df["A10_blank"] * df["A_social"]
    df["Qchat_x_Family"] = df["A10_blank"] * df["Family_ASD"]
    df["A10_x_SocialIssues"]        = df["A10_blank"]         * df["Social_Issues"]



    df["Is_Toddler"]    = (df["Age_Years"] <= 3).astype(int)
    df["Is_Child"]      = ((df["Age_Years"] > 3) & (df["Age_Years"] <= 11)).astype(int)
    df["Is_Adolescent"] = (df["Age_Years"] > 11).astype(int)
    df["Age_Group_Ord"] = (
        df["Is_Toddler"] * 0 + df["Is_Child"] * 1 + df["Is_Adolescent"] * 2
    )



    df["Clinical_Flag_Count"] = (
        df["Jaundice"]          + df["Family_ASD"]       + df["Speech_Delay"]
        + df["Learning_Disorder"] + df["Genetic"]        + df["Social_Issues"]
        + df["Global_Dev_Delay"]  + df["Anxiety"]        + df["Depression"]
    )

    return df



FEATURE_COLS = [
    "A1", "A2", "A3", "A4", "A5",
    "A6", "A7", "A8", "A9", "A10",


    "Age_Years", "Sex_M",


    "Ethnicity_WE", "Ethnicity_SA", "Ethnicity_ME",


    "Jaundice", "Family_ASD",


    "Speech_Delay", "Learning_Disorder", "Genetic",
    "Depression", "Global_Dev_Delay", "Social_Issues", "Anxiety",


    "Completed_HCP",


    "A_comm",
    "A_social",
    "A_imaginative",
    "A_sensory",
    "A10_blank",


    "Qchat_x_Social",
    "Qchat_x_Family",
    "Speech_x_Learning",
    "Genetic_x_GDD",
    "Anxiety_x_Dep",
    "SocialDef_x_SpeechDelay",
    "SocialDef_x_SocialIssues",
    "A10_x_SocialIssues",


    "Is_Toddler", "Is_Child", "Is_Adolescent", "Age_Group_Ord",



    "Clinical_Flag_Count",
]



def build_ensemble():
    xgb = XGBClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=350, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    gbt = GradientBoostingClassifier(
        n_estimators=220, max_depth=5, learning_rate=0.06,
        subsample=0.8, min_samples_split=4, random_state=42,
    )

    lr = LogisticRegression(
        C=1.0, max_iter=1500, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )

    return xgb, rf, gbt, lr



def evaluate(model, X_test, y_test, label="Model") -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "label":     label,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_test, y_prob),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }

def print_metrics(m: dict):
    print(f"\n  {'─'*54}")
    print(f"  {m['label']}")
    print(f"  {'─'*54}")
    print(f"  Accuracy  : {m['accuracy']*100:.2f}%")
    print(f"  Precision : {m['precision']*100:.2f}%")
    print(f"  Recall    : {m['recall']*100:.2f}%")
    print(f"  F1 Score  : {m['f1']*100:.2f}%")
    print(f"  ROC AUC   : {m['auc']*100:.2f}%")

    if "training_time" in m:
        print(f"  Time      : {m['training_time']:.2f} sec")


def plot_model_comparison(results):
    names = []
    acc, prec, rec, f1, auc = [], [], [], [], []

    for name, obj in results.items():
        m = obj["metrics"]
        names.append(name)
        acc.append(m["accuracy"])
        prec.append(m["precision"])
        rec.append(m["recall"])
        f1.append(m["f1"])
        auc.append(m["auc"])

    x = np.arange(len(names))
    width = 0.16

    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    b1 = plt.bar(x - 2*width, acc, width, label="Accuracy")
    b2 = plt.bar(x - width, prec, width, label="Precision")
    b3 = plt.bar(x, rec, width, label="Recall")
    b4 = plt.bar(x + width, f1, width, label="F1")
    b5 = plt.bar(x + 2*width, auc, width, label="AUC")

    def add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.002,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90
            )

    add_labels(b1)
    add_labels(b2)
    add_labels(b3)
    add_labels(b4)
    add_labels(b5)

    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Score")

    plt.ylim(0.85, 1.0)

    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.title("Model Performance Comparison")

    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_plot("model_comparison")

def reduce_features(X, y, feature_cols, variance_threshold=0.01, corr_threshold=0.92):
    df_feat = pd.DataFrame(X, columns=feature_cols)


    sel = VarianceThreshold(threshold=variance_threshold)
    sel.fit(X)
    kept = [f for f, s in zip(feature_cols, sel.get_support()) if s]
    X = df_feat[kept].values


    corr_matrix = pd.DataFrame(X, columns=kept).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    kept = [f for f in kept if f not in to_drop]
    df_feat = df_feat[kept]
    X = df_feat.values

    print(f"  ▸ Features after filter stage: {len(kept)} (from {len(feature_cols)})")
    return X, kept

def select_by_permutation(pipeline, X_test, y_test, feature_cols, top_n=25):
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10, random_state=42, scoring="f1"
    )
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print("\n  Top features by permutation importance:")
    print(imp_df.head(top_n).to_string(index=False))


    useful = imp_df[imp_df["importance_mean"] > 0]["feature"].tolist()
    print(f"\n  ▸ Dropping {len(feature_cols) - len(useful)} low-value features")
    return useful

def train(paths: dict, out_dir: str = MODEL_OUT_DIR):

    df_raw = load_and_merge(paths)

    print("\n  ▸ Engineering features...")
    df = engineer_features(df_raw)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df["ASD"].values

    X, feature_cols = reduce_features(X, y, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  ▸ Train set : {len(X_train)} samples")
    print(f"  ▸ Test  set : {len(X_test)} samples")

    xgb_m, rf_m, gbt_m, lr_m = build_ensemble()

    def make_pipeline(model):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", model)
        ])

    # Tuning XGBoot
    print("\n  🔧 Tuning XGBoost...")

    pipe = make_pipeline(xgb_m)
    xgb_param_dist = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.2],
    }

    xgb_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=xgb_param_dist,
        n_iter=20,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    xgb_search.fit(X_train, y_train)

    xgb_m = xgb_search.best_estimator_.named_steps["model"]

    print("  ✅ Best XGB params:", xgb_search.best_params_)

    # Tuning Random Forest
    print("\n  🔧 Tuning Random Forest...")

    pipe = make_pipeline(rf_m)
    rf_param_dist = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [8, 10, 12, 15],
        "model__min_samples_split": [2, 4, 6],
        "model__min_samples_leaf": [1, 2, 3],
    }

    rf_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=rf_param_dist,
        n_iter=15,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    rf_search.fit(X_train, y_train)

    rf_m  = rf_search.best_estimator_.named_steps["model"]

    print("  ✅ Best RF params:", rf_search.best_params_)

    # 🔧 Tuning Gradient Boosting
    print("\n  🔧 Tuning Gradient Boosting...")
    pipe = make_pipeline(gbt_m)
    gbt_param_dist = {
        "model__n_estimators": [150, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 5],
        "model__min_samples_split": [2, 4, 6],
        "model__subsample": [0.7, 0.8, 1.0],
    }

    gbt_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=gbt_param_dist,
        n_iter=15,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    gbt_search.fit(X_train, y_train)

    gbt_m = gbt_search.best_estimator_.named_steps["model"]

    print("  ✅ Best GBT params:", gbt_search.best_params_)
    

    print("\n" + "═" * 64)
    print("  TRAINING ENSEMBLE(XGB + RF + GBT + LR)")
    print("═" * 64)

    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_m),
            ("rf", rf_m),
            ("gbt", gbt_m),
        ],
        voting="soft",
        weights=[4, 3, 3],
    )

    model_results = {}

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", ensemble)
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    ens_time = time.time() - start_time
    
    ens_metrics = evaluate(pipeline, X_test, y_test, "Ensemble")
    ens_metrics["training_time"] = ens_time

    model_results["Ensemble"] = {
        "model": pipeline,
        "metrics": ens_metrics
    }

    print_metrics(ens_metrics)

    # -------------------------
    # INDIVIDUAL MODELS
    # -------------------------

    for name, mdl in [
        ("XGBoost", xgb_m),
        ("Random Forest", rf_m),
        ("Gradient Boosting", gbt_m),
        ("Logistic Regression", lr_m),
    ]:
        pipe = make_pipeline(mdl)

        start = time.time()
        pipe.fit(X_train, y_train)
        t = time.time() - start

        metrics = evaluate(pipe, X_test, y_test, name)
        metrics["training_time"] = t

        model_results[name] = {
            "model": pipe,
            "metrics": metrics
        }

        print_metrics(metrics)


    # -------------------------
    # COMBINATIONS
    # -------------------------
    combos = {
        "XGB+RF": [("xgb", xgb_m), ("rf", rf_m)],
        "XGB+GBT": [("xgb", xgb_m), ("gbt", gbt_m)],
        "RF+GBT": [("rf", rf_m), ("gbt", gbt_m)],
    }

    for name, est in combos.items():
        print(f"\nTraining {name}...")

        combo = VotingClassifier(estimators=est, voting="soft")

        pipe = make_pipeline(combo)

        start = time.time()
        pipe.fit(X_train, y_train)
        t = time.time() - start

        metrics = evaluate(pipe, X_test, y_test, name)
        metrics["training_time"] = t

        model_results[name] = {"model": pipe, "metrics": metrics}

        print_metrics(metrics)
    plot_model_comparison(model_results)

    # -------------------------
    # SELECT BEST MODEL
    # -------------------------
    def score(m):
        return (
            0.4 * m["recall"] +
            0.3 * m["f1"] +
            0.3 * m["auc"]
        )

    best_model_name = max(
        model_results,
        key=lambda k: score(model_results[k]["metrics"])
    )

    best_entry = model_results[best_model_name]
    final_pipeline = best_entry["model"]
    best_metrics = best_entry["metrics"]

    print(f"\n🏆 Best Model: {best_model_name}")

    print("\n" + "═" * 64)
    print("  PERMUTATION FEATURE SELECTION")
    print("═" * 64)

    useful_features = select_by_permutation(
        final_pipeline, X_test, y_test, feature_cols, top_n=25
    )

    useful_idx = [feature_cols.index(f) for f in useful_features]

    X_train = X_train[:, useful_idx]
    X_test  = X_test[:, useful_idx]
    X       = X[:, useful_idx]

    feature_cols = useful_features


    final_pipeline.fit(X_train, y_train)


    final_metrics_r = evaluate(
        final_pipeline, X_test, y_test,
        f"{best_model_name} [reduced]"
    )
    print_metrics(final_metrics_r)


    print("\n  Confusion Matrix:")

    best_preds = final_pipeline.predict(X_test)

    cm = confusion_matrix(y_test, best_preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No ASD", "ASD"],
        yticklabels=["No ASD", "ASD"]
    )

    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(f"confusion_matrix_{best_model_name}")

    print("\n  Full classification report:")
    print(classification_report(y_test, best_preds,
                                target_names=["No ASD", "ASD"]))


    print("\n" + "═" * 64)
    print("  5-FOLD CROSS-VALIDATION")
    print("═" * 64)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for metric_name in ["accuracy", "f1", "roc_auc"]:
        scores = cross_val_score(
            final_pipeline,
            X,
            y,
            cv=cv,
            scoring=metric_name,
            n_jobs=-1
        )
        fold_str = ", ".join(f"{s:.4f}" for s in scores)
        print(f"  {metric_name:<12}: {scores.mean():.4f} ± {scores.std():.4f} [{fold_str}]")

    cv_acc = cross_val_score(final_pipeline, X, y, cv=cv, scoring="accuracy")

    os.makedirs(out_dir, exist_ok=True)

    trained_model = final_pipeline.named_steps["model"]
    try:
        if isinstance(trained_model, VotingClassifier):
            estimators = dict(trained_model.named_estimators_)

            importances = []

            if "xgb" in estimators:
                importances.append(estimators["xgb"].feature_importances_)
            if "rf" in estimators:
                importances.append(estimators["rf"].feature_importances_)

            if importances:
                avg_imp = np.mean(importances, axis=0)

                fi_df = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": avg_imp
                }).sort_values("importance", ascending=False)

                fi_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
                print("✅ Feature importance saved (ensemble avg)")
            else:
                print("⚠ No compatible models for feature importance in ensemble")

        elif hasattr(trained_model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": trained_model.feature_importances_
            }).sort_values("importance", ascending=False)

            fi_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
            print("✅ Feature importance saved (single model)")

        else:
            print("⚠ Model does not support feature importance")

    except Exception as e:
        print(f"⚠ Feature importance failed: {e}")

    def plot_feature_importance(csv_path, top_n=20):
        df = pd.read_csv(csv_path).head(top_n)

        plt.figure(figsize=(8,6))
        sns.barplot(x="importance", y="feature", data=df)
        plt.title("Top Feature Importance")
        plt.tight_layout()
        save_plot("feature_importance")

    plot_feature_importance(os.path.join(out_dir, "feature_importance.csv"))

    best_metrics = final_metrics_r

    model_bundle = {
        "model":        final_pipeline,
        "model_used":   best_model_name,
        "feature_cols": feature_cols,
        "metrics": {
            "accuracy":  float(final_metrics_r["accuracy"]),
            "precision": float(final_metrics_r["precision"]),
            "recall":    float(final_metrics_r["recall"]),
            "f1":        float(final_metrics_r["f1"]),
            "auc":       float(final_metrics_r["auc"]),
            "cv_mean":   float(cv_acc.mean()),
            "cv_std":    float(cv_acc.std()),
        },
        "dataset_info": {
            "total_samples": int(len(df)),
            "asd_positive":  int(df["ASD"].sum()),
            "asd_negative":  int((df["ASD"] == 0).sum()),
            "sources_used":  list(paths.keys()),
            "n_features":    len(feature_cols),
        },
    }

    pkl_path = os.path.join(out_dir, "asd_model.pkl")
    try:
        joblib.dump(model_bundle, pkl_path)
    except Exception as e:
        print(f"❌ Failed saving model: {e}")


    metrics_for_api = {
        "accuracy":       round(final_metrics_r["accuracy"]  * 100, 2),
        "precision":      round(final_metrics_r["precision"] * 100, 2),
        "recall":         round(final_metrics_r["recall"]    * 100, 2),
        "f1":             round(final_metrics_r["f1"]        * 100, 2),
        "auc":            round(final_metrics_r["auc"]       * 100, 2),
        "cv_mean":        round(float(cv_acc.mean())     * 100, 2),
        "cv_std":         round(float(cv_acc.std())      * 100, 2),
        "n_features":     len(feature_cols),
        "n_samples":      int(len(df)),
    }

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_for_api, f, indent=2)

    print(f"\n  ✓ Model saved  → {pkl_path}")
    print(f"  ✓ Metrics JSON → {json_path}")
    print("\n" + "═" * 64)
    print("  TRAINING COMPLETE")
    print("═" * 64 + "\n")

    return model_bundle


def parse_args():
    p = argparse.ArgumentParser(
        description="Train ASD ensemble model on three age-stratified datasets."
    )
    p.add_argument("--toddler",     default=DEFAULT_PATHS["toddler"])
    p.add_argument("--early-child", default=DEFAULT_PATHS["early_child"])
    p.add_argument("--adolescent",  default=DEFAULT_PATHS["adolescent"])
    p.add_argument("--no-toddler",    action="store_true")
    p.add_argument("--no-adolescent", action="store_true")
    p.add_argument("--out-dir",     default=MODEL_OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    active_paths = {"early_child": args.early_child}
    if not args.no_toddler:
        active_paths["toddler"]    = args.toddler
    if not args.no_adolescent:
        active_paths["adolescent"] = args.adolescent

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║        ASD ENSEMBLE MODEL — TRAINING PIPELINE            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Datasets to load : {list(active_paths.keys())}")
    print(f"  Output directory : {args.out_dir}\n")

    train(paths=active_paths, out_dir=args.out_dir)
