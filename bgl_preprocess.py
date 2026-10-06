import os
import pandas as pd
from shared_steps import add_templates, group_into_sessions

BGL_LOG_FILE = "BGL.log"
OUTPUT_FOLDER = "processed"
WINDOW_MINUTES = 30
REAL_SEVERITY_LEVELS = ["INFO", "WARNING", "SEVERE", "ERROR", "FATAL", "FAILURE"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# 1. Load the log file
# read every line into a pandas Series (one row = one log line)
with open(BGL_LOG_FILE, encoding="utf-8", errors="replace") as f:
    raw_lines = pd.Series(f.read().splitlines())
print("Lines read:", len(raw_lines))

# remove non-printable characters (a few lines in the file are corrupted)
raw_lines = raw_lines.str.replace(r"[^\x20-\x7e]", "", regex=True)


# 2. Split each line into columns
# a BGL line looks like:
# - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
# the message is everything after the 9th space, so split at most 9 times
df = raw_lines.str.split(" ", n=9, expand=True)
df.columns = ["LabelToken", "Timestamp", "Date", "Node", "Time",
              "NodeRepeat", "Type", "Component", "Level", "Content"]
df["LineId"] = range(1, len(df) + 1)
df["Content"] = df["Content"].fillna("").str.strip()

# Fix lines with no node ID
# some lines have "-" as the node and no repeated node, e.g.
# - 1119415930 2005.06.21 - 2005-06-21-21.52.10.214285 RAS KERNEL FATAL Kill job 20251 timed out. Block freed.
# so everything after it is one column to the left. Move those columns back
missing_node = (df["Node"] == "-") & (df["NodeRepeat"] == "RAS")
df.loc[missing_node, "Content"] = (df.loc[missing_node, "Level"].fillna("") + " " + df.loc[missing_node, "Content"]).str.strip()
df.loc[missing_node, "Level"] = df.loc[missing_node, "Component"]
df.loc[missing_node, "Component"] = df.loc[missing_node, "Type"]
df.loc[missing_node, "Type"] = df.loc[missing_node, "NodeRepeat"]
df.loc[missing_node, "NodeRepeat"] = "-"
print("Lines without a node ID (fixed):", missing_node.sum())

# Labels
# first word "-" = normal (0), anything else = alert (1)
# the alert type (e.g. KERNDTLB) is saved in Subtype
df["Label"] = (df["LabelToken"] != "-").astype(int)
df["Subtype"] = df["LabelToken"].where(df["Label"] == 1, "")


# 3. Remove invalid rows
# keep a line only if the timestamp is a number and the level is a real severity
is_well_formed = (df["Timestamp"].str.isdigit().fillna(False)
                  & df["Level"].isin(REAL_SEVERITY_LEVELS))
print("Malformed lines dropped:", (~is_well_formed).sum())
df = df[is_well_formed].copy()
df["Timestamp"] = df["Timestamp"].astype(int)


# 4. Make sessions
# BGL has no block ID like HDFS, so we use 30 minute time windows instead
# SessionId = which window the line falls in (0, 1, 2 ...)
seconds_since_start = df["Timestamp"] - df["Timestamp"].min()
df["SessionId"] = seconds_since_start // (WINDOW_MINUTES * 60)

# templates and session grouping are the same for both datasets (see shared_steps.py)
df, template_table = add_templates(df, id_prefix="B")
session_table = group_into_sessions(df, dataset_name="BGL")


# 5. Save the results
df[["LineId", "Timestamp", "Node", "Level", "Component", "Content",
    "EventId", "EventTemplate", "Label", "Subtype", "SessionId"]].to_csv(
    f"{OUTPUT_FOLDER}/bgl_structured.csv", index=False)
template_table.to_csv(f"{OUTPUT_FOLDER}/bgl_templates.csv", index=False)
session_table.to_json(f"{OUTPUT_FOLDER}/bgl_sessions.jsonl", orient="records", lines=True)


# 6. Print a summary
print("Lines kept:", len(df))
print("Empty messages:", (df["Content"] == "").sum())
print("Alert lines:", df["Label"].sum(), f"({df['Label'].mean():.2%})")
print("Distinct templates:", len(template_table))

print("\nTop alert categories:")
print(df.loc[df["Label"] == 1, "Subtype"].value_counts().head(10))

# check how well severity matches the label
print("\nSeverity vs label (0 = normal, 1 = alert):")
print(pd.crosstab(df["Level"], df["Label"]))

print(f"\nSessions ({WINDOW_MINUTES} minute windows):", len(session_table))
print("Anomalous sessions:", session_table["label"].sum(),
      f"({session_table['label'].mean():.2%})")
print("Lines per session (median):", session_table["events"].str.len().median())