"""Train and evaluate the anomaly classifiers and cluster the anomalous sessions.

Input : <dataset>_features.csv from feature_extraction.py
        (columns SessionId, label, subtype, start_ts + feature columns f_ / evt_ / hdfs_)
Output: result tables (CSV), error-analysis samples, cluster profile and the saved model.

Example (run from the scripts folder):
    python train_models.py --features ../HDFS_v1/processed/hdfs_features.csv \
        --sessions ../HDFS_v1/processed/hdfs_sessions.jsonl --out results_hdfs
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is only needed for Table III
    XGBClassifier = None

SEED = 42
FEATURE_PREFIXES = ("f_", "evt_", "hdfs_")  # same prefixes as feature_extraction.py


# ----------------------------------------------------------------------------
# 1. Loading and splitting
# ----------------------------------------------------------------------------
def load_features(path, drop_cols):
    """Read the feature table and return it with the list of feature columns."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    # optional ablation, e.g. drop severity features that nearly encode the BGL label
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    return df, feature_cols


def time_split(df, train_frac=0.70, val_frac=0.15):
    """Chronological split by session start time: the test set is the latest sessions."""
    df = df.sort_values("start_ts", kind="stable").reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    n_val = int(len(df) * val_frac)
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def random_split(df, train_frac=0.70):
    """Stratified random split with the same 70/15/15 sizes, for comparison."""
    train, rest = train_test_split(df, train_size=train_frac, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    return train, val, test


def describe_split(name, parts):
    """Print size and anomaly rate of each part, to be reported in the paper."""
    for part_name, part in zip(("train", "validation", "test"), parts):
        print(f"[{name}] {part_name:<10} sessions={len(part):>8}  anomalous={int(part['label'].sum()):>6}"
              f" ({part['label'].mean():.2%})")


def stratified_sample(X, y, n):
    """Return at most n rows, keeping the anomaly rate (used for slow steps such as kNN and CV)."""
    if len(X) <= n:
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=SEED)
    return X_s, y_s


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def make_random_forest():
    # class_weight balanced: the anomalous class is rare, so its errors count more
    return Pipeline([("model", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED))])


def make_xgboost(y_train):
    # scale_pos_weight = normal / anomalous in the training data (same idea as class_weight)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    return Pipeline([("model", XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=pos_weight,
        tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED))])


def make_isolation_forest(y_train):
    # unsupervised: contamination is set to the anomaly rate of the training data
    contamination = float(np.clip(y_train.mean(), 0.001, 0.5))
    return IsolationForest(n_estimators=200, contamination=contamination, n_jobs=-1, random_state=SEED)


def taught_models():
    """The four classifiers taught in the unit, used for the preliminary comparison (Table II)."""
    return {
        "Logistic regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(
            class_weight="balanced", max_iter=1000))]),
        "k-nearest neighbours": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1))]),
        "Decision tree": Pipeline([("model", DecisionTreeClassifier(
            class_weight="balanced", random_state=SEED))]),
        "Random forest": make_random_forest(),
    }


# ----------------------------------------------------------------------------
# 3. Scoring and metrics
# ----------------------------------------------------------------------------
def anomaly_scores(model, X):
    """Higher score = more anomalous, for both classifiers and Isolation Forest."""
    if isinstance(model, IsolationForest):
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]


def best_threshold(y_val, scores):
    """Threshold with the highest F1 on the validation set (never chosen on the test set)."""
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def score_metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "average_precision": average_precision_score(y_true, scores),  # threshold-free
        "threshold": threshold,
    }


def cv_f1(model, X, y, n_sample):
    """Mean and standard deviation of F1 over 5 stratified folds (on a sample, for speed)."""
    X_s, y_s = stratified_sample(X, y, n_sample)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X_s, y_s, cv=folds, scoring="f1")
    return scores.mean(), scores.std()


# ----------------------------------------------------------------------------
# 4. Experiments
# ----------------------------------------------------------------------------
def preliminary_comparison(train, val, feature_cols, args):
    """Table II: the four taught classifiers, scored on the validation set."""
    X_tr, y_tr = train[feature_cols], train["label"]
    X_va, y_va = val[feature_cols], val["label"]
    rows = []
    for name, model in taught_models().items():
        # kNN stores every training row, so it is fitted on a sample
        X_fit, y_fit = stratified_sample(X_tr, y_tr, args.knn_sample) if "neighbours" in name else (X_tr, y_tr)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_va)
        mean, std = cv_f1(model, X_tr, y_tr, args.cv_sample)
        rows.append({"model": name,
                     "precision": precision_score(y_va, pred, zero_division=0),
                     "recall": recall_score(y_va, pred, zero_division=0),
                     "f1": f1_score(y_va, pred, zero_division=0),
                     "cv_f1_mean": mean, "cv_f1_sd": std})
        print("Table II row:", rows[-1])
    return pd.DataFrame(rows)


def compare_main_models(split_name, parts, feature_cols, args):
    """Table III: random forest (baseline), XGBoost and Isolation Forest on the same test set."""
    train, val, test = parts
    X_tr, y_tr = train[feature_cols], train["label"]
    X_va, y_va = val[feature_cols], val["label"]
    X_te, y_te = test[feature_cols], test["label"]

    models = {"Random forest (baseline)": make_random_forest()}
    if XGBClassifier is not None:
        models["XGBoost"] = make_xgboost(y_tr)
    else:
        print("xgboost is not installed: skipping XGBoost (conda install -c conda-forge xgboost)")
    models["Isolation Forest"] = make_isolation_forest(y_tr)

    rows, fitted = [], {}
    for name, model in models.items():
        if isinstance(model, IsolationForest):
            model.fit(X_tr)                      # no labels used for fitting
            cv = (np.nan, np.nan)
        else:
            model.fit(X_tr, y_tr)
            cv = cv_f1(model, X_tr, y_tr, args.cv_sample)
        val_scores = anomaly_scores(model, X_va)
        threshold = best_threshold(y_va, val_scores)
        test_scores = anomaly_scores(model, X_te)
        row = {"split": split_name, "model": name, "cv_f1_mean": cv[0], "cv_f1_sd": cv[1],
               "val_f1": score_metrics(y_va, val_scores, threshold)["f1"]}
        row.update(score_metrics(y_te, test_scores, threshold))
        rows.append(row)
        fitted[name] = (model, threshold, test_scores)
        print("Table III row:", row)
    return pd.DataFrame(rows), fitted


def error_analysis(name, threshold, scores, test, sessions_path, out_dir, n=10):
    """Sample false positives / false negatives and attach their event sequences for manual reading."""
    pred = (scores >= threshold).astype(int)
    test = test.assign(score=scores, pred=pred)
    fp = test[(test.pred == 1) & (test.label == 0)]
    fn = test[(test.pred == 0) & (test.label == 1)]
    sample = pd.concat([fp.sample(min(n, len(fp)), random_state=SEED).assign(error="false positive"),
                        fn.sample(min(n, len(fn)), random_state=SEED).assign(error="false negative")])
    keep = ["SessionId", "error", "label", "score", "f_length", "f_duration"]
    sample = sample[[c for c in keep if c in sample.columns]]
    print(f"Error analysis for {name}: {len(fp)} false positives, {len(fn)} false negatives in the test set")

    if sessions_path and os.path.exists(sessions_path):
        wanted, found = set(sample["SessionId"]), []
        for chunk in pd.read_json(sessions_path, lines=True, chunksize=100000):
            found.append(chunk[chunk["SessionId"].isin(wanted)][["SessionId", "events"]])
        events = pd.concat(found)
        events["events"] = events["events"].apply(lambda e: " ".join(e))
        sample = sample.merge(events, on="SessionId", how="left")
    sample.to_csv(os.path.join(out_dir, "error_analysis.csv"), index=False)


# ----------------------------------------------------------------------------
# 5. Clustering inside the anomalous class (labels are NOT used to fit)
# ----------------------------------------------------------------------------
def cluster_anomalies(df, feature_cols, args):
    anomalous = df[df["label"] == 1].reset_index(drop=True)   # one class only
    X = anomalous[feature_cols]
    X_scaled = StandardScaler().fit_transform(X)
    print(f"Clustering {len(anomalous)} anomalous sessions with {len(feature_cols)} features")

    # k from 2 to 8: inertia (elbow) and silhouette (separation)
    results = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_, sample_size=min(5000, len(X_scaled)), random_state=SEED)
        results[k] = (km, sil)
        print(f"k={k}  inertia={km.inertia_:.0f}  silhouette={sil:.3f}  sizes={np.bincount(km.labels_).tolist()}")
    # default rule: best silhouette among the k whose smallest cluster holds at least
    # --min-cluster-share of the sessions (a cluster of a few sessions is an outlier group,
    # not a failure type). If no k qualifies, fall back to the best silhouette.
    valid = [key for key, (model, _) in results.items()
             if np.bincount(model.labels_).min() / len(anomalous) >= args.min_cluster_share]
    if not valid:
        print("Warning: every k has a very small cluster, using the best silhouette instead")
        valid = list(results)
    k = args.k or max(valid, key=lambda key: results[key][1])
    km = results[k][0]
    print("Chosen k:", k)

    # describe each cluster: mean vs the class mean, in standard deviations
    class_mean, class_std = X.mean(), X.std().replace(0, np.nan)
    evt_cols = [c for c in feature_cols if c.startswith("evt_")]
    profile = []
    for c in range(k):
        members = anomalous[km.labels_ == c]
        z = ((members[feature_cols].mean() - class_mean) / class_std).fillna(0)
        # a missing template (count 0) is often the strongest anomaly signal
        cluster_missing, class_missing = (members[evt_cols] == 0).mean(), (anomalous[evt_cols] == 0).mean()
        missing = (cluster_missing - class_missing).sort_values(ascending=False)
        # real example: the session closest to the cluster centre
        centre_dist = ((X_scaled[km.labels_ == c] - km.cluster_centers_[c]) ** 2).sum(axis=1)
        profile.append({
            "cluster": c, "sessions": len(members),
            "higher_than_class": "; ".join(f"{f} ({z[f]:+.1f})" for f in z.sort_values(ascending=False).index[:2]),
            "lower_than_class": "; ".join(f"{f} ({z[f]:+.1f})" for f in z.sort_values().index[:2]),
            # share of cluster sessions without the template, versus the whole anomalous class
            "more_often_missing": "; ".join(f"{f} ({cluster_missing[f]:.0%} vs {class_missing[f]:.0%})" for f in missing.index[:3]),
            "example_session": members.iloc[int(np.argmin(centre_dist))]["SessionId"],
            "label_share_anomalous": members["label"].mean(),   # only a sanity check
        })
    profile = pd.DataFrame(profile)

    # DBSCAN as a check on the K-means result: eps from the 5-nearest-neighbour distances
    kdist = NearestNeighbors(n_neighbors=5).fit(X_scaled).kneighbors(X_scaled)[0][:, -1]
    eps = args.eps or float(np.percentile(kdist, 90))
    db = DBSCAN(eps=eps, min_samples=5).fit(X_scaled)
    print(f"DBSCAN check: eps={eps:.2f}, clusters={len(set(db.labels_) - {-1})}, "
          f"noise={np.mean(db.labels_ == -1):.1%}")
    return profile


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="path to <dataset>_features.csv")
    parser.add_argument("--sessions", help="path to <dataset>_sessions.jsonl (adds event sequences to the error analysis)")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--drop-cols", default="", help="comma-separated feature columns to leave out (ablation)")
    parser.add_argument("--knn-sample", type=int, default=50000, help="training rows used to fit kNN")
    parser.add_argument("--cv-sample", type=int, default=50000, help="rows used for the 5-fold cross-validation")
    parser.add_argument("--k", type=int, help="number of clusters (default: best silhouette without tiny clusters)")
    parser.add_argument("--min-cluster-share", type=float, default=0.01,
                        help="smallest allowed cluster as a share of the sessions when choosing k")
    parser.add_argument("--only-clustering", action="store_true", help="skip the classifiers and only cluster")
    parser.add_argument("--eps", type=float, help="DBSCAN eps (default: from k-distance)")
    parser.add_argument("--skip-random", action="store_true", help="skip the random-split comparison")
    parser.add_argument("--skip-clustering", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, feature_cols = load_features(args.features, [c for c in args.drop_cols.split(",") if c])
    print(f"{len(df)} sessions, {len(feature_cols)} features, anomaly rate {df['label'].mean():.2%}")
    print(f"An always-normal model would reach {1 - df['label'].mean():.2%} accuracy and catch nothing")

    if args.only_clustering:
        profile = cluster_anomalies(df, feature_cols, args)
        profile.to_csv(os.path.join(args.out, "table_IV_clusters.csv"), index=False)
        print(profile.to_string())
        return

    # time-based split is the main evaluation; the random split shows how much easier it is
    time_parts = time_split(df)
    describe_split("time", time_parts)

    table2 = preliminary_comparison(time_parts[0], time_parts[1], feature_cols, args)
    table2.to_csv(os.path.join(args.out, "table_II_preliminary.csv"), index=False)

    table3, fitted = compare_main_models("time", time_parts, feature_cols, args)
    if not args.skip_random:
        random_parts = random_split(df)
        describe_split("random", random_parts)
        table3_random, _ = compare_main_models("random", random_parts, feature_cols, args)
        table3 = pd.concat([table3, table3_random])
    table3.to_csv(os.path.join(args.out, "table_III_main.csv"), index=False)

    # error analysis and the saved model use the best supervised model on the validation F1
    supervised = {k: v for k, v in fitted.items() if k != "Isolation Forest"}
    best_name = max(supervised, key=lambda k: table3[(table3.split == "time") & (table3.model == k)]["val_f1"].iloc[0])
    model, threshold, scores = fitted[best_name]
    error_analysis(best_name, threshold, scores, time_parts[2], args.sessions, args.out)
    joblib.dump({"pipeline": model, "features": feature_cols, "threshold": threshold, "name": best_name},
                os.path.join(args.out, "model.joblib"))
    print("Saved model:", best_name)

    if not args.skip_clustering:
        profile = cluster_anomalies(df, feature_cols, args)
        profile.to_csv(os.path.join(args.out, "table_IV_clusters.csv"), index=False)
        print(profile.to_string())


if __name__ == "__main__":
    main()
