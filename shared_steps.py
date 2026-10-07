# Shared steps used by both HDFS and BGL
# Each dataset script makes a table with these columns first:
# LineId, Timestamp, Level, Component, Content, Label, Subtype, SessionId
# then calls add_templates() and group_into_sessions() below

import pandas as pd

# shared severity scale so severity means the same thing in both datasets
# BGL has 6 levels and HDFS mostly INFO/WARN, so we group them like syslog does (RFC 5424)
# 0 = info, 1 = warning, 2 = error or worse
SEVERITY_SCORE = {
        "INFO": 0,
        "WARN": 1, "WARNING": 1,
        "ERROR": 2, "SEVERE": 2, "FATAL": 2, "FAILURE": 2,
}


def add_templates(df, id_prefix):
        # Replace the parts of each message that change (numbers, paths, hex)
        # with <*> so the same type of message gets the same template
        # e.g. "8 floating point alignment exceptions" -> "<*> floating point alignment exceptions"

        # HDFS block IDs, e.g. blk_38865049064139660 or blk_-6952295868487656571
        # masked first, otherwise a negative ID is only half masked ("blk_-<*>") and the
        # same message type gets a different template for positive and negative IDs
        masked = df["Content"].str.replace(r"blk_-?\d+", "<*>", regex=True)
        # file paths, e.g. /bgl/apps/test.rts or ./mmcs_db_server
        masked = masked.str.replace(r"\S*/\S*/\S*|(?<!\S)\.?/\S*", "<*>", regex=True)
        # long hex values with no digits, e.g. ffffffff
        masked = masked.str.replace(r"\b[0-9a-fA-F]{8,}\b", "<*>", regex=True)
        # any word with a digit in it (same idea as the Drain parser, He et al. 2017)
        masked = masked.str.replace(r"\S*\d\S*", "<*>", regex=True)

        # lines with no message get their own template
        df["EventTemplate"] = masked.where(masked != "", "<EMPTY>")

        # give each template an ID, e.g. B1, B2, B3 ...
        template_numbers, _ = pd.factorize(df["EventTemplate"])
        df["EventId"] = [id_prefix + str(n + 1) for n in template_numbers]

        # count how many times each template appears and how often it is an anomaly
        template_table = df.groupby(["EventId", "EventTemplate"]).agg(
                Occurrences=("Label", "size"),
                AnomalyShare=("Label", "mean"),
        ).reset_index()
        template_table = template_table.sort_values("Occurrences", ascending=False)

        return df, template_table


def group_into_sessions(logs, dataset_name):
        # Put all lines with the same SessionId into one session
        # A session is labelled anomaly (1) if any of its lines is an anomaly

        # keep the lines in the same order as the original log
        logs = logs.sort_values("LineId")

        # convert each level to the 0/1/2 scale (unknown levels count as info)
        logs["SeverityScore"] = logs["Level"].map(SEVERITY_SCORE).fillna(0).astype(int)

        session_table = logs.groupby("SessionId").agg(
                start_ts=("Timestamp", "min"),
                end_ts=("Timestamp", "max"),
                events=("EventId", list),
                levels=("Level", list),
                severities=("SeverityScore", list),
                components=("Component", list),
                line_ids=("LineId", list),
                label=("Label", "max"),
                subtype=("Subtype", lambda tags: sorted(set(tags) - {""})),
        ).reset_index()

        # note which dataset the session came from
        session_table.insert(0, "source", dataset_name)

        return session_table