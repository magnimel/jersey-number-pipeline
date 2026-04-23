# Individual STR Swap: CRNN for Jersey Number Recognition

Branch: `individual_task_beichen`

This branch integrates an alternative scene text recognition (STR) backend into the jersey-number pipeline. The baseline team pipeline uses fine-tuned PARSeq; this branch adds CRNN as a lighter CTC-based STR model and keeps the same torso-crop input and tracklet-level voting flow.

The original project documentation is preserved in [README_original.md](README_original.md).

## Model Choice

Chosen model: `CRNN` (Convolutional Recurrent Neural Network)

Source code: STRHub/PARSeq model hub vendored in this repository under `str/parseq/strhub/models/crnn`.

Pretrained weights: `pretrained=crnn`, downloaded by STRHub from the public `baudm/parseq` release URL listed in `str/parseq/strhub/models/utils.py`.

Pretraining data: the STRHub/PARSeq release documents standard STR LMDB datasets including MJSynth and SynthText in `str/parseq/Datasets.md`.

Why CRNN: CRNN is a standard, compact STR baseline with convolutional features, bidirectional recurrent sequence modeling, and CTC decoding. It is usually faster and simpler than PARSeq, but less context-aware and likely less robust on low-resolution, occluded jersey crops.

## Code Changes

The branch adds CRNN support without removing the PARSeq baseline.

- `main.py` adds `--str_backend {parseq,crnn}`, `--str_checkpoint`, `--str_batch_size`, `--str_epochs`, and `--str_train_batch_size`.
- `configuration.py` adds `crnn_str_model: pretrained=crnn` for SoccerNet and Hockey.
- `str.py` now detects CTC models and decodes CRNN outputs into numeric jersey labels with at most two digits.
- `str.py` writes optional timing metrics via `--metrics_file`.
- `helpers.py` treats `0` through `99` as valid jersey-number predictions for the alternative STR path.
- `main.py` falls back to heuristic voting for CRNN because the improved BiLSTM aggregator was trained on PARSeq-shaped logits.
- `evaluate.py` can write combined accuracy and timing metrics JSON.
- `scripts/extract_str_loss_curve.py` exports STRHub TensorBoard loss curves to CSV.
- `scripts/make_evalai_submission.py` normalizes final predictions and optionally zips them for upload.

## Setup

Clone the team repository and create the required branch:

```bash
git clone https://github.com/AJAR-of-Cookies/jersey-number-recognition-team-5.git
cd jersey-number-recognition-team-5
git checkout -b individual_task_beichen
```

Install dependencies and download model weights/data using the team setup flow:

```bash
python3 setup.py SoccerNet
python3 scripts/download_data.py
```

This local checkout did not have `conda`, SoccerNet data, Hockey data, or model checkpoints available, so full fine-tuning and evaluation were not executed here. The commands below are the reproducible run plan for a prepared workstation or Colab.

## Fine-Tuning

Fine-tune CRNN briefly on the Hockey jersey-number LMDB:

```bash
python3 main.py Hockey train \
  --train_str \
  --str_backend crnn \
  --str_epochs 5 \
  --str_train_batch_size 128
```

Fine-tune CRNN on weakly labeled SoccerNet jersey-number crops:

```bash
python3 main.py SoccerNet train \
  --train_str \
  --str_backend crnn \
  --str_epochs 10 \
  --str_train_batch_size 128
```

Training checkpoints are written under `str/parseq/outputs/crnn/<run>/checkpoints/`. Use the best `*.ckpt` path as `--str_checkpoint` for evaluation.

Export loss curves after each run:

```bash
python3 scripts/extract_str_loss_curve.py \
  --run_dir str/parseq/outputs/crnn/<run> \
  --output reports/crnn_loss_curve.csv
```

Record the final training loss and validation loss from the exported CSV in the results table below.

## Evaluation

Run CRNN on the same SoccerNet test pipeline used for the PARSeq baseline:

```bash
python3 main.py SoccerNet test \
  --str_backend crnn \
  --str_checkpoint str/parseq/outputs/crnn/<run>/checkpoints/<best>.ckpt \
  --str_batch_size 512
```

If torso crops and intermediate files already exist, resume from the last completed stage:

```bash
python3 main.py SoccerNet test \
  --resume \
  --str_backend crnn \
  --str_checkpoint str/parseq/outputs/crnn/<run>/checkpoints/<best>.ckpt \
  --str_batch_size 512
```

Calculate Top-1 Accuracy and combine it with STR timing:

```bash
python3 evaluate.py \
  --pred out/SoccerNetResults/final_results.json \
  --gt data/SoccerNet/test/test_gt.json \
  --timing out/SoccerNetResults/test/crnn_str_metrics.json \
  --output reports/crnn_test_metrics.json
```

Run the challenge split and prepare an upload artifact:

```bash
python3 main.py SoccerNet challenge \
  --str_backend crnn \
  --str_checkpoint str/parseq/outputs/crnn/<run>/checkpoints/<best>.ckpt \
  --str_batch_size 512

python3 scripts/make_evalai_submission.py \
  --pred out/SoccerNetResults/challenge_final_results.json \
  --output reports/crnn_challenge_submission.zip
```

For the test split, package predictions similarly if EvalAI accepts local test submissions:

```bash
python3 scripts/make_evalai_submission.py \
  --pred out/SoccerNetResults/final_results.json \
  --output reports/crnn_test_submission.zip
```

## Results

The PARSeq baseline is the team's replicated result. CRNN results must be filled after running the commands above on the prepared dataset/checkpoint environment.

| Model | Fine-tuning | Top-1 Accuracy | STR Speed | Notes |
|---|---:|---:|---:|---|
| PARSeq baseline | Team replicated | 87.6% | Not recorded here | Reference baseline |
| CRNN pretrained | None | Pending | Pending | `pretrained=crnn`, numeric CTC decoding |
| CRNN Hockey -> SoccerNet | 5 Hockey epochs + 5-10 SoccerNet epochs | Pending | Pending | Expected final comparison row |

Loss-curve fields to record after training:

| Run | Dataset | Epochs | Final Train Loss | Final Val Loss | CSV |
|---|---:|---:|---:|---:|---|
| CRNN Hockey | Hockey LMDB | 5 | Pending | Pending | `reports/crnn_hockey_loss_curve.csv` |
| CRNN SoccerNet | SoccerNet LMDB | 5-10 | Pending | Pending | `reports/crnn_soccer_loss_curve.csv` |

## Analysis

CRNN should be faster and cheaper to fine-tune than PARSeq because it has a comparatively small CNN plus recurrent sequence head and uses greedy CTC decoding. That makes it a useful baseline for limited compute and for measuring whether the jersey-number task needs PARSeq's stronger language/context modeling.

The likely disadvantage is accuracy. Jersey crops are often low-resolution, motion-blurred, partially occluded, and limited to one or two digits. PARSeq's transformer decoder and learned sequence modeling can use stronger positional/context cues, while CRNN's CTC path can emit repeated, blank, or non-digit tokens. This branch mitigates that by filtering predictions to digits and reusing the team's tracklet-level heuristic voting, but the improved BiLSTM aggregator should be retrained before comparing it fairly with CRNN logits.

## Timeline

Using April 23, 2026 as the start date, the 3-week target date is May 14, 2026.
