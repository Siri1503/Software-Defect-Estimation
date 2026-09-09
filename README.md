# Software Defect Estimation

A Flask-based machine learning application for estimating whether a software module is likely to be defective using software complexity and code metrics.

## Project overview

The application follows the workflow described in the project report: load a dataset, inspect the data, preprocess it, train multiple classifiers, compare model performance, and predict whether a software module is defective.

The included implementation uses a generated software-metrics dataset by default, so the project runs without requiring the original dataset. You can also upload your own CSV with the expected columns.

## Features

- Upload and preview a software-defect CSV dataset
- Automatically generate a demo dataset when no CSV is present
- Validate required feature columns
- Preprocess numeric software metrics
- Perform a stratified 80/20 train/test split
- Train and evaluate multiple classifiers
- Predict defective vs non-defective software modules
- Responsive Flask dashboard inspired by the output screens in the original project documentation

## Models

The application supports:

- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Extra Trees
- XGBoost
- CatBoost
- LightGBM
- Hybrid soft-voting ensemble

XGBoost, CatBoost, and LightGBM are shown when those libraries are installed.

## Input metrics

The prediction form uses 16 software metrics:

`loc`, `v(g)`, `ev(g)`, `iv(g)`, `l`, `d`, `i`, `e`, `IOCode`, `IOComment`, `IOBlank`, `Unique OP`, `Unique OPND`, `Total OP`, `Total OPND`, `Branch Count`

The target column is:

`defect`

where `0` means non-defective and `1` means defective.

## Run locally

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Application workflow

```text
Dataset
   ↓
Upload / Demo Generation
   ↓
Data Preview
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Accuracy Evaluation
   ↓
Manual Software-Metric Input
   ↓
Defective / Non-Defective Prediction
```

## Notes

The original source code was not available when this repository was rebuilt. This implementation was recreated from the project documentation and output screens, so it is a new working implementation consistent with the documented functionality rather than a copy of the original code.

Accuracy values are calculated from the current dataset and are not hardcoded to match historical screenshots.

## Tech stack

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- XGBoost
- CatBoost
- LightGBM
- HTML/CSS

## Author

Divya Sree Yellampalli
