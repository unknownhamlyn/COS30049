# Charts for the report (Data Analysis section)
# Reads the processed files the other scripts produce, so it works without the
# full raw logs. Run from the scripts folder:  python make_figures.py

import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "../figures"
HDFS_FOLDER = "../HDFS_v1/processed"
BGL_FOLDER = "../BGL/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# 1. Class balance, HDFS vs BGL
# Anomalies are a small minority in both datasets, which is why the report uses
# F1 (not accuracy): predicting "all normal" would already score ~97% on HDFS.
hdfs_sessions = pd.read_json(f"{HDFS_FOLDER}/hdfs_sessions.jsonl", lines=True)
bgl_sessions = pd.read_json(f"{BGL_FOLDER}/bgl_sessions.jsonl", lines=True)

balance = pd.DataFrame({
    "HDFS": [(hdfs_sessions["label"] == 0).sum(), (hdfs_sessions["label"] == 1).sum()],
    "BGL": [(bgl_sessions["label"] == 0).sum(), (bgl_sessions["label"] == 1).sum()],
}, index=["Normal", "Anomalous"])

plt.figure(figsize=(6, 4))
balance.plot(kind="bar", ax=plt.gca())
plt.title("Class balance: HDFS vs BGL")
plt.xlabel("Session type")
plt.ylabel("Number of sessions")
plt.xticks(rotation=0)
plt.yscale("log")  # normal dwarfs anomalous, so log scale keeps both visible
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/fig1_class_balance.png", dpi=200)
plt.close()


# 2. The "missing step" pattern (HDFS)
# anomaly often shows up as an expected event that never happens. 
# Here we compare, for normal vs anomalous sessions, the share that
# never log a "stored" step or never log a "received" step.
hdfs = pd.read_csv(f"{HDFS_FOLDER}/hdfs_features.csv")

# hdfs_missing_stored is already 1 when a block was written but never stored.
# For "never received" we use the receiving-minus-received gap being positive.
hdfs["missing_received"] = (hdfs["hdfs_receiving_minus_received"] > 0).astype(int)

missing_step = pd.DataFrame({
    "Never stored": [
        hdfs.loc[hdfs["label"] == 0, "hdfs_missing_stored"].mean(),
        hdfs.loc[hdfs["label"] == 1, "hdfs_missing_stored"].mean(),
    ],
    "Never received": [
        hdfs.loc[hdfs["label"] == 0, "missing_received"].mean(),
        hdfs.loc[hdfs["label"] == 1, "missing_received"].mean(),
    ],
}, index=["Normal", "Anomalous"]) * 100

plt.figure(figsize=(6, 4))
missing_step.plot(kind="bar", ax=plt.gca())
plt.title("Sessions missing an expected step (HDFS)")
plt.xlabel("Session type")
plt.ylabel("Share of sessions (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/fig2_missing_step.png", dpi=200)
plt.close()


# 3. Feature vs label strength (HDFS)
# Point-biserial correlation is just Pearson correlation between each numeric
# feature and the 0/1 label. This is the "statistical method" and it shows which
# features carry the signal, before any model is trained.
feature_cols = [c for c in hdfs.columns if c.startswith(("f_", "evt_", "hdfs_"))]
correlations = hdfs[feature_cols].corrwith(hdfs["label"]).abs().sort_values(ascending=False)
top15 = correlations.head(15).sort_values()  # sort so the biggest bar is on top

plt.figure(figsize=(6, 5))
top15.plot(kind="barh", ax=plt.gca())
plt.title("Feature correlation with the label (HDFS, top 15)")
plt.xlabel("Absolute correlation with label")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/fig3_feature_correlation.png", dpi=200)
plt.close()


# 4. BGL severity vs label
# Nearly every alert is FATAL, but most FATAL lines are normal, so severity
# points to where alerts cluster without identifying them. This is also why the
# BGL score drops when the severity features are removed.
bgl = pd.read_csv(f"{BGL_FOLDER}/bgl_structured.csv")
severity_vs_label = pd.crosstab(bgl["Level"], bgl["Label"])
severity_vs_label.columns = ["Normal", "Alert"]
level_order = ["INFO", "WARNING", "SEVERE", "ERROR", "FAILURE", "FATAL"]
severity_vs_label = severity_vs_label.reindex(
    [lvl for lvl in level_order if lvl in severity_vs_label.index])

plt.figure(figsize=(6, 4))
severity_vs_label.plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("BGL severity vs label")
plt.xlabel("Severity level")
plt.ylabel("Number of lines")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/fig4_severity_vs_label.png", dpi=200)
plt.close()


# 5. Print a summary
print("Saved figures to", OUTPUT_FOLDER)
for name in ["fig1_class_balance", "fig2_missing_step",
                "fig3_feature_correlation", "fig4_severity_vs_label"]:
    print(" -", name + ".png")