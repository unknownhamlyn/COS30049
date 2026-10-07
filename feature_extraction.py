# Feature extraction for HDFS and BGL
# Turns each session (from *_sessions.jsonl) into one row of numbers
# Run from the scripts folder:  python feature_extraction.py
#
# Column name prefixes, so later scripts can pick feature groups easily:
#   f_     shared features, mean the same thing in both datasets (used for cross-system test)
#   evt_   count of each common event template in the session
#   hdfs_  HDFS block workflow checks (only for HDFS)
# SessionId, label, subtype and start_ts are kept for splitting and checking,
# they are NOT features

import os
import math
from collections import Counter
import pandas as pd

# where each dataset's processed files are (change these if your folders differ)
DATASETS = {
    "hdfs": "../HDFS_v1/processed",
    "bgl": "../BGL/processed",
}

# templates that together cover this share of all lines get their own column,
# everything rarer is counted together in f_rare_event_share
COMMON_TEMPLATE_COVERAGE = 0.99


def entropy(counts):
    # Shannon entropy of the event counts in a session
    # low = the same event repeated over and over, high = many different events
    total = sum(counts.values())
    return abs(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def count_matching(events, templates, keyword):
    # count events in the session whose template text contains the keyword
    return sum(1 for e in events if keyword in templates.get(e, ""))


for name, folder in DATASETS.items():
    sessions_file = f"{folder}/{name}_sessions.jsonl"
    templates_file = f"{folder}/{name}_templates.csv"
    if not os.path.exists(sessions_file):
        print(f"Skipping {name}: {sessions_file} not found")
        continue

    print(f"\n{name.upper()}")
    sessions = pd.read_json(sessions_file, lines=True)
    template_table = pd.read_csv(templates_file)
    print("Sessions:", len(sessions))

    # template ID -> template text, used for the HDFS workflow checks
    template_text = dict(zip(template_table["EventId"], template_table["EventTemplate"]))

    # pick the common templates (most frequent first, until 99% of lines are covered)
    template_table = template_table.sort_values("Occurrences", ascending=False)
    coverage = template_table["Occurrences"].cumsum() / template_table["Occurrences"].sum()
    common_events = template_table.loc[coverage.shift(fill_value=0) < COMMON_TEMPLATE_COVERAGE, "EventId"].tolist()
    common_set = set(common_events)
    print("Common templates (own column):", len(common_events), "of", len(template_table))

    rows = []
    for s in sessions.itertuples(index=False):
        events = s.events
        counts = Counter(events)
        n = len(events)
        severities = s.severities
        duration = s.end_ts - s.start_ts

        row = {
            "SessionId": s.SessionId,
            "label": s.label,
            "subtype": ";".join(s.subtype),
            "start_ts": s.start_ts,

            # 1. volume and repetition (He et al. 2016)
            "f_length": n,
            "f_unique_events": len(counts),
            "f_unique_ratio": len(counts) / n,
            "f_event_entropy": entropy(counts),

            # 2. timing: a stalled or retried session takes longer
            "f_duration": duration,
            "f_events_per_second": n / (duration + 1),

            # 3. severity on the shared 0/1/2 scale (see shared_steps.py)
            "f_warning_share": severities.count(1) / n,
            "f_error_share": severities.count(2) / n,
            "f_max_severity": max(severities),

            # 4. how many different parts of the system were involved
            "f_unique_components": len(set(s.components)),

            # 5. rare events: messages outside the common templates
            #    unusual messages are often the anomalous ones
            "f_rare_event_share": sum(c for e, c in counts.items() if e not in common_set) / n,
        }

        # 6. count of each common template (the "bag of events", Xu et al. 2009)
        for e in common_events:
            row["evt_" + e] = counts.get(e, 0)

        # 7. HDFS block workflow checks (invariants, Lou et al. 2010)
        # a normal block is received, acknowledged and stored the same number of times,
        # so a non-zero difference means a step in the workflow went missing
        if name == "hdfs":
            receiving = count_matching(events, template_text, "Receiving block")
            received = count_matching(events, template_text, "Received block")
            responder_done = count_matching(events, template_text, "terminating")
            stored = count_matching(events, template_text, "addStoredBlock")
            row["hdfs_receiving_minus_received"] = receiving - received
            row["hdfs_receiving_minus_responder"] = receiving - responder_done
            row["hdfs_received_minus_stored"] = received - stored
            # block was written to but never registered as stored
            row["hdfs_missing_stored"] = int(receiving > 0 and stored == 0)
            row["hdfs_exception_events"] = count_matching(events, template_text, "xception")

        rows.append(row)

    features = pd.DataFrame(rows)
    out_file = f"{folder}/{name}_features.csv"
    features.to_csv(out_file, index=False)

    feature_cols = [c for c in features.columns if c.startswith(("f_", "evt_", "hdfs_"))]
    print("Feature columns:", len(feature_cols))
    print("Anomalous sessions:", features["label"].sum(), f"({features['label'].mean():.2%})")
    print("Saved", out_file)
