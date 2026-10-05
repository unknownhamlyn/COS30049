''''
Shared steps for both HDFS and BGL.

Input table columns:
        LineId, Timestamp, Level, Component, Content, Label, Subtype, SessionId
        - Label: 0 = normal, 1 = anomaly
        - Subtype: anomaly category, or "" if none
        - SessionId: block ID (HDFS) or time window (BGL)

Call add_templates() then group_into_sessions() so both datasets
end up in the same format.
'''''

import pandas as pd


def add_templates(df, id_prefix):
        # Replace the parts of each message that change (numbers, paths, hex)
        # with <*> so the same type of message gets the same template
        # e.g. "8 floating point alignment exceptions" -> "<*> floating point alignment exceptions"

        # file paths, e.g. /bgl/apps/test.rts or ./mmcs_db_server
        masked = df["Content"].str.replace(r"\S*/\S*/\S*|(?<!\S)\.?/\S*", "<*>", regex=True)
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

        session_table = logs.groupby("SessionId").agg(
                start_ts=("Timestamp", "min"),
                end_ts=("Timestamp", "max"),
                events=("EventId", list),
                levels=("Level", list),
                components=("Component", list),
                line_ids=("LineId", list),
                label=("Label", "max"),
                subtype=("Subtype", lambda tags: sorted(set(tags) - {""})),
        ).reset_index()

        # note which dataset the session came from
        session_table.insert(0, "source", dataset_name)

        return session_table