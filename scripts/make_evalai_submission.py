#!/usr/bin/env python3
"""Create a normalized SoccerNet jersey-number prediction file.

The pipeline writes final results as {tracklet_id: jersey_number}. This helper
validates that shape, normalizes values to integers in [-1, 99], and optionally
zips the JSON for upload workflows that require an archive.
"""

import argparse
import json
import os
import zipfile


def normalize_predictions(predictions):
    normalized = {}
    for tracklet_id, value in predictions.items():
        try:
            pred = int(value)
        except (TypeError, ValueError):
            pred = -1
        if pred < -1 or pred > 99:
            pred = -1
        normalized[str(tracklet_id)] = pred
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Prepare EvalAI-style prediction JSON for SoccerNet jersey numbers.")
    parser.add_argument("--pred", required=True,
                        help="Pipeline final_results JSON, e.g. out/SoccerNetResults/challenge_final_results.json")
    parser.add_argument("--output", required=True,
                        help="Output .json or .zip path")
    parser.add_argument("--json_name", default="predictions.json",
                        help="JSON filename to place inside a zip archive")
    args = parser.parse_args()

    with open(args.pred, "r") as f:
        predictions = normalize_predictions(json.load(f))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.output.endswith(".zip"):
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(args.json_name, json.dumps(predictions, indent=2))
        print(f"Wrote {len(predictions)} predictions to {args.output}:{args.json_name}")
    else:
        with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
