"""Print the event sequence of chosen sessions with the template text.

Used to read false positives / false negatives and the example session of each cluster.
Repeated consecutive events are collapsed ("3 x Receiving block ...").

Examples (run from the scripts folder):
    python show_sessions.py blk_-1470250012612845302 blk_8351506178782871227
    python show_sessions.py --ids-from results_hdfs/error_analysis.csv
"""

import argparse

import pandas as pd

DEFAULT_SESSIONS = "../HDFS_v1/processed/hdfs_sessions.jsonl"
DEFAULT_TEMPLATES = "../HDFS_v1/processed/hdfs_templates.csv"


def collapse(events):
    """Turn ['H1','H1','H5'] into [('H1', 2), ('H5', 1)]."""
    runs = []
    for event in events:
        if runs and runs[-1][0] == event:
            runs[-1][1] += 1
        else:
            runs.append([event, 1])
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_ids", nargs="*", help="SessionId values to print")
    parser.add_argument("--ids-from", help="CSV with a SessionId column (for example error_analysis.csv)")
    parser.add_argument("--sessions", default=DEFAULT_SESSIONS)
    parser.add_argument("--templates", default=DEFAULT_TEMPLATES)
    args = parser.parse_args()

    wanted = set(args.session_ids)
    if args.ids_from:
        wanted |= set(pd.read_csv(args.ids_from)["SessionId"].astype(str))
    if not wanted:
        parser.error("give at least one SessionId or --ids-from")

    templates = pd.read_csv(args.templates)
    text = dict(zip(templates["EventId"], templates["EventTemplate"]))

    # the sessions file is large, so it is read in chunks and only the wanted rows are kept
    for chunk in pd.read_json(args.sessions, lines=True, chunksize=100000):
        for row in chunk[chunk["SessionId"].astype(str).isin(wanted)].itertuples():
            print(f"\n{row.SessionId}  ({len(row.events)} events, label {row.label}, "
                  f"{row.end_ts - row.start_ts} s)")
            for event, count in collapse(row.events):
                print(f"  {count:>3} x {event:<4} {text.get(event, '?')}")


if __name__ == "__main__":
    main()
