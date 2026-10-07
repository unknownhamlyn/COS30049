# COS30049 Assignment 2 - Log Anomaly Detection (AI4Cyber)

Detects anomalous sessions in system logs using classification, and groups the
anomalies into sub-types using clustering. Built and evaluated on two datasets
(HDFS and BGL) with one shared pipeline.

## Datasets

The raw logs are not committed (too large, see .gitignore). Download them from
Loghub's Zenodo record and place each .log file as shown below.

- **HDFS_v1** (base dataset, provided): https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1
  - IMPORTANT: place at `HDFS_v1/HDFS.log`, and the label file at `HDFS_v1/processed/anomaly_label.csv`
- **BGL** (additional dataset): https://zenodo.org/records/8196385/files/BGL.zip?download=1
  - place at `BGL/BGL.log`

## Setup (venv + pip)
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib xgboost

On macOS, xgboost also needs the OpenMP runtime:
brew install libomp

## How to run

Run from the `scripts` folder, in this order (each step uses the previous one's output):
python hdfs_preprocess.py # clean + parse HDFS, build sessions
python bgl_preprocess.py # clean + parse BGL, build sessions
python feature_extraction.py # turn sessions into feature tables
python train_models.py --features ../HDFS_v1/processed/hdfs_features.csv
--sessions ../HDFS_v1/processed/hdfs_sessions.jsonl --out results_hdfs
python make_figures.py # figures for the Data Analysis section

Results (tables, model, error analysis, figures) are written under `results_hdfs/`
and `figures/`.

## Scripts

- `hdfs_preprocess.py` / `bgl_preprocess.py` - dataset-specific cleaning, labelling and session building
- `shared_steps.py` - template masking and session grouping shared by both datasets
- `feature_extraction.py` - one row of numeric features per session
- `train_models.py` - train/evaluate the classifiers, cluster the anomalies, save the model and figures
- `make_figures.py` - the Data Analysis figures
- `show_sessions.py` - helper to print a session's event sequence (for reading false positives/negatives)

## Team

Group 7, Session 21
Members: Oscar Thomas, William Truong, Janet Yin
Tutor: Yinwei Bao
