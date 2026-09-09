from pathlib import Path
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_PATH = UPLOAD_DIR / "current_dataset.csv"

FEATURES = [
    "loc", "v(g)", "ev(g)", "iv(g)", "l", "d", "i", "e",
    "IOCode", "IOComment", "IOBlank", "Unique OP", "Unique OPND",
    "Total OP", "Total OPND", "Branch Count"
]
TARGET = "defect"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


def generate_demo_dataset(rows=1200):
    X, y = make_classification(
        n_samples=rows,
        n_features=len(FEATURES),
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        weights=[0.68, 0.32],
        class_sep=1.0,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=FEATURES)
    for col in FEATURES:
        frame[col] = np.abs(frame[col] * 20).round(3)
    frame[TARGET] = y
    return frame


def load_dataset():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if TARGET in df.columns and all(c in df.columns for c in FEATURES):
            return df
    df = generate_demo_dataset()
    df.to_csv(DATA_PATH, index=False)
    return df


def available_models():
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": Pipeline([("scale", StandardScaler()), ("svc", SVC(probability=True, random_state=42))]),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42),
    }
    if XGBClassifier:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42
        )
    if CatBoostClassifier:
        models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=42)
    if LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(random_state=42, verbosity=-1)

    hybrid_members = [
        ("rf", RandomForestClassifier(n_estimators=180, random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=180, random_state=42)),
    ]
    models["Hybrid Model"] = VotingClassifier(estimators=hybrid_members, voting="soft")
    return models


def split_data(df):
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@app.route("/")
def home():
    df = load_dataset()
    return render_template("index.html", rows=len(df), features=len(FEATURES))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Please upload a CSV file.", "error")
            return redirect(url_for("upload"))
        try:
            df = pd.read_csv(file)
            missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
            if missing:
                flash("Missing required columns: " + ", ".join(missing), "error")
                return redirect(url_for("upload"))
            df.to_csv(DATA_PATH, index=False)
            session.pop("selected_model", None)
            flash(f"Dataset uploaded successfully ({len(df)} rows).", "success")
            return redirect(url_for("view_dataset"))
        except Exception as exc:
            flash(f"Could not read dataset: {exc}", "error")
    return render_template("upload.html", features=FEATURES, target=TARGET)


@app.route("/reset-demo")
def reset_demo():
    df = generate_demo_dataset()
    df.to_csv(DATA_PATH, index=False)
    session.clear()
    flash("Demo software-metrics dataset restored.", "success")
    return redirect(url_for("view_dataset"))


@app.route("/view")
def view_dataset():
    df = load_dataset()
    preview = df.head(30).to_html(classes="data-table", index=False, border=0)
    return render_template("view.html", table=preview, rows=len(df), cols=len(df.columns))


@app.route("/preprocess")
def preprocess():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    stats = {
        "rows": len(df),
        "missing": int(df[FEATURES + [TARGET]].isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "train": len(X_train),
        "test": len(X_test),
        "defective": int(df[TARGET].sum()),
        "non_defective": int((df[TARGET] == 0).sum()),
    }
    return render_template("preprocess.html", stats=stats)


@app.route("/models", methods=["GET", "POST"])
def models():
    model_map = available_models()
    result = None
    if request.method == "POST":
        selected = request.form.get("model")
        if selected not in model_map:
            flash("Please select a valid model.", "error")
        else:
            df = load_dataset()
            X_train, X_test, y_train, y_test = split_data(df)
            model = model_map[selected]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result = {
                "name": selected,
                "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
                "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            }
            session["selected_model"] = selected
            flash(f"{selected} trained successfully.", "success")
    return render_template("models.html", model_names=list(model_map.keys()), result=result)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    selected = session.get("selected_model", "Extra Trees")
    model_map = available_models()
    if selected not in model_map:
        selected = "Extra Trees"

    if request.method == "POST":
        try:
            values = [float(request.form.get(c, 0)) for c in FEATURES]
            df = load_dataset()
            X_train, _, y_train, _ = split_data(df)
            model = model_map[selected]
            model.fit(X_train, y_train)
            input_df = pd.DataFrame([values], columns=FEATURES)
            pred = int(model.predict(input_df)[0])
            prediction = "Defective Software Module" if pred == 1 else "Non-Defective Software Module"
            if hasattr(model, "predict_proba"):
                probability = round(float(model.predict_proba(input_df)[0][pred]) * 100, 2)
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "error")

    return render_template("predict.html", features=FEATURES, selected_model=selected,
                           prediction=prediction, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
