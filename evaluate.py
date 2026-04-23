import json
import argparse
import sys

def evaluate(prediction_path, ground_truth_path):
    # 1. Load Ground Truth
    try:
        with open(ground_truth_path, 'r') as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    # 2. Load Student Predictions
    try:
        with open(prediction_path, 'r') as f:
            pred_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {prediction_path}")
        sys.exit(1)
    correct = 0
    total = 0
    missing = 0

    # 3. Compare
    for uuid, gt_number in gt_data.items():
        total += 1
        # Check if the student made a prediction for this UUID
        if uuid in pred_data:
            pred_number = pred_data[uuid]
            # Ensure we compare integers (handle potential string inputs)
            try:
                if int(pred_number) == int(gt_number):
                    correct += 1
            except ValueError:
                # If prediction is not a valid number, it counts as incorrect
                pass
        else:
            missing += 1
    
    # 4. Results
    accuracy = correct / total if total > 0 else 0
    
    metrics = {
        "total_samples": total,
        "predictions_provided": total - missing,
        "correct_predictions": correct,
        "missing_predictions": missing,
        "accuracy": accuracy,
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Jersey Number Recognition")
    parser.add_argument("--pred", required=True, help="Path to your predictions JSON file")
    parser.add_argument("--gt", required=True, help="Path to the test_gt.json file")
    parser.add_argument("--timing", default=None,
                        help="Optional STR timing JSON produced by str.py --metrics_file")
    parser.add_argument("--output", default=None,
                        help="Optional path to write combined metrics JSON")
    args = parser.parse_args()
    metrics = evaluate(args.pred, args.gt)
    if args.timing:
        try:
            with open(args.timing, 'r') as f:
                metrics["timing"] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: timing file not found at {args.timing}")

    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"Total Samples (GT): {metrics['total_samples']}")
    print(f"Predictions Provided: {metrics['predictions_provided']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print(f"Missing Predictions: {metrics['missing_predictions']}")
    print("-" * 30)
    print(f"FINAL ACCURACY: {metrics['accuracy']:.2%}")
    if "timing" in metrics:
        timing = metrics["timing"]
        avg = timing.get("avg_inference_seconds_per_image")
        throughput = timing.get("throughput_images_per_second")
        if avg is not None:
            print(f"AVG STR INFERENCE: {avg:.6f}s/image")
        if throughput is not None:
            print(f"STR THROUGHPUT: {throughput:.2f} images/s")
    print("-" * 30)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
