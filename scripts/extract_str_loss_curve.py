#!/usr/bin/env python3
"""Export STR training loss curves from a PARSeq/STRHub TensorBoard run."""

import argparse
import csv
import glob
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Export STR loss curves to CSV.")
    parser.add_argument("--run_dir", required=True,
                        help="Training output directory, e.g. str/parseq/outputs/crnn/<date_time>")
    parser.add_argument("--output", required=True,
                        help="CSV path to write: tag,step,value")
    parser.add_argument("--tags", nargs="*", default=["loss", "val_loss"],
                        help="TensorBoard scalar tags to export")
    args = parser.parse_args()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard is required: pip install tensorboard", file=sys.stderr)
        return 2

    event_files = glob.glob(os.path.join(args.run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"No TensorBoard event files found under {args.run_dir}", file=sys.stderr)
        return 1

    rows = []
    for event_file in event_files:
        accumulator = EventAccumulator(event_file)
        accumulator.Reload()
        available_tags = set(accumulator.Tags().get("scalars", []))
        for tag in args.tags:
            if tag not in available_tags:
                continue
            for event in accumulator.Scalars(tag):
                rows.append({
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                    "event_file": event_file,
                })

    if not rows:
        print(f"No requested scalar tags found. Requested: {', '.join(args.tags)}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "step", "value", "event_file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scalar points to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
