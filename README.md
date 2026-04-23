# Individual STR Swap: CRNN for Jersey Number Recognition

Branch: `individual_task_beichen`

For my individual task, I added a CRNN scene text recognition model to the jersey-number pipeline. The team pipeline already uses PARSeq, so this branch keeps PARSeq and adds CRNN as another option. CRNN still uses the same torso crops and the same tracklet voting idea at the end.

The original project README is still saved in [README_original.md](README_original.md).

## Model Choice

Model: `CRNN` (Convolutional Recurrent Neural Network)

Code location: `str/parseq/strhub/models/crnn`

Weights: `pretrained=crnn`, downloaded through STRHub from the public `baudm/parseq` release listed in `str/parseq/strhub/models/utils.py`.

Training data source: STRHub lists the usual STR datasets like MJSynth and SynthText in `str/parseq/Datasets.md`.

I chose CRNN because it is a simple and common STR baseline. It is smaller than PARSeq and uses CTC decoding, so it should be faster to run and easier to fine-tune. The downside is that it may not be as accurate on blurry or blocked jersey numbers.

## What Changed

- Added `--str_backend {parseq,crnn}` so the pipeline can switch between PARSeq and CRNN.
- Added CRNN options for checkpoints, batch size, epochs, and training batch size.
- Added CRNN config entries for SoccerNet and Hockey.
- Updated `str.py` so it can decode CTC output from CRNN into one- or two-digit jersey numbers.
- Added optional timing metrics with `--metrics_file`.
- Allowed jersey numbers from `0` to `99` for the CRNN path.
- Used heuristic voting for CRNN because the improved BiLSTM aggregator was trained for PARSeq logits.
- Updated `evaluate.py` so it can save accuracy and timing metrics together.
- Added helper scripts for exporting loss curves and creating EvalAI submission files.

## Setup

Clone the team repo and create the branch:

```bash
git clone https://github.com/magnimel/jersey-number-pipeline.git
git checkout -b individual_task_beichen
```

Install dependencies and download the data:

```bash
python3 setup.py SoccerNet
python3 scripts/download_data.py
```

I could not run the full training or testing locally because this checkout did not have `conda`, the SoccerNet/Hockey data, or the model checkpoints. The commands below show how to run everything on a prepared machine or Colab.

## Fine-Tuning

Fine-tune CRNN on the Hockey jersey-number LMDB:

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

Checkpoints are saved under `str/parseq/outputs/crnn/<run>/checkpoints/`. Use the best `*.ckpt` file as `--str_checkpoint` when testing.

Export the loss curve after training:

```bash
python3 scripts/extract_str_loss_curve.py \
  --run_dir str/parseq/outputs/crnn/<run> \
  --output reports/crnn_loss_curve.csv
```

Use the CSV to fill in the training loss and validation loss in the results table.

## Evaluation

Run CRNN on the SoccerNet test split:

```bash
python3 main.py SoccerNet test \
  --str_backend crnn \
  --str_checkpoint str/parseq/outputs/crnn/<run>/checkpoints/<best>.ckpt \
  --str_batch_size 512
```

If the crop files already exist, resume from the last finished step:

```bash
python3 main.py SoccerNet test \
  --resume \
  --str_backend crnn \
  --str_checkpoint str/parseq/outputs/crnn/<run>/checkpoints/<best>.ckpt \
  --str_batch_size 512
```

Calculate Top-1 Accuracy and combine it with timing:

```bash
python3 evaluate.py \
  --pred out/SoccerNetResults/final_results.json \
  --gt data/SoccerNet/test/test_gt.json \
  --timing out/SoccerNetResults/test/crnn_str_metrics.json \
  --output reports/crnn_test_metrics.json
```

Run the challenge split and create the upload zip:

```bash
python3 main.py SoccerNet challenge \
  --str_backend crnn \
  --str_checkpoint str/parseq/outputs/crnn/<run>/checkpoints/<best>.ckpt \
  --str_batch_size 512

python3 scripts/make_evalai_submission.py \
  --pred out/SoccerNetResults/challenge_final_results.json \
  --output reports/crnn_challenge_submission.zip
```

For the test split, the same script can package the predictions:

```bash
python3 scripts/make_evalai_submission.py \
  --pred out/SoccerNetResults/final_results.json \
  --output reports/crnn_test_submission.zip
```

## Results

The PARSeq baseline is the team's replicated result. The CRNN rows should be filled in after running the commands above with the full data and checkpoints.

| Model | Fine-tuning | Top-1 Accuracy | STR Speed | Notes |
|---|---:|---:|---:|---|
| PARSeq baseline | Team replicated | 87.6% | Not recorded here | Reference baseline |
| CRNN pretrained | None | Pending | Pending | `pretrained=crnn`, numeric CTC decoding |
| CRNN Hockey -> SoccerNet | 5 Hockey epochs + 5-10 SoccerNet epochs | Pending | Pending | Final comparison row |

Loss-curve fields to fill in after training:

| Run | Dataset | Epochs | Final Train Loss | Final Val Loss | CSV |
|---|---:|---:|---:|---:|---|
| CRNN Hockey | Hockey LMDB | 5 | Pending | Pending | `reports/crnn_hockey_loss_curve.csv` |
| CRNN SoccerNet | SoccerNet LMDB | 5-10 | Pending | Pending | `reports/crnn_soccer_loss_curve.csv` |

## Analysis

CRNN should be faster and cheaper to fine-tune than PARSeq because it is a smaller model and uses greedy CTC decoding. This makes it useful as a simpler baseline for checking whether the full PARSeq model is really needed for jersey numbers.

The main risk is lower accuracy. Jersey crops can be blurry, small, blocked by other players, or only one or two digits long. PARSeq has stronger sequence modeling, while CRNN can repeat characters, output blanks, or miss digits. To reduce this, this branch filters outputs to numbers and keeps the team's tracklet-level voting. A fair comparison would also retrain the BiLSTM aggregator for CRNN logits.
