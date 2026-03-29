main_env = 'jersey'  # conda env that runs main.py and esrgan.py

pose_home = 'pose/ViTPose'
pose_env = 'vitpose'

str_home = 'str/parseq/'
str_env = 'parseq2'
str_platform = 'cu113'

# centroids
reid_env = 'centroids'
reid_script = 'centroid_reid.py'

# digit classifier (Python 3.11, movinets)
digit_env = 'digit_classifier'

# aggregation (Python 3.11, pytorch-lightning, torchmetrics)
agg_env = 'aggregation'

reid_home = 'reid/'

# ---------------------------------------------------------------------------
# Real-ESRGAN configuration
# ---------------------------------------------------------------------------
# Path to the RealESRGAN_x4plus.pth checkpoint (downloaded by setup.py).
esrgan_model = 'models/RealESRGAN_x4plus.pth'
# Upscaling factor – must match the model (4 for RealESRGAN_x4plus).
esrgan_scale = 4
# Tile size for patch-based inference; 0 = whole image (needs more VRAM).
# Set to 256 or 512 if you run out of GPU memory on small GPUs.
esrgan_tile = 0
# Use FP16 inference for speed on modern GPUs (Ampere / Turing and above).
esrgan_half = False
# Number of images processed per GPU forward pass. On OOM the pipeline halves
# this automatically until it finds a size that fits, so setting it high is safe.
esrgan_batch_size = 128
# ---------------------------------------------------------------------------


dataset = {'SoccerNet':
                {'root_dir': './data/SoccerNet',
                 'working_dir': './out/SoccerNetResults',
                 'test': {
                        'images': 'test/images',
                        'gt': 'test/test_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/test',
                        'illegible_result': 'illegible.json',
                        'soccer_ball_list': 'soccer_ball.json',
                        'sim_filtered': 'test/main_subject_0.4.json',
                        'gauss_filtered': 'test/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'legible.json',
                        'raw_legible_result': 'raw_legible_resnet34.json',
                        'pose_input_json': 'pose_input.json',
                        'pose_output_json': 'pose_results.json',
                        'crops_folder': 'crops',
                        'crops_sr_folder': 'test/crops_sr',
                        'jersey_id_result': 'jersey_id_results.json',
                        'digit_predictions': 'test/digit_predictions.json',
                        'final_result': 'final_results.json'
                    },
                 'val': {
                        'images': 'val/images',
                        'gt': 'val/val_gt.json',
                        'feature_output_folder': 'out/SoccerNetResults/val',
                        'illegible_result': 'illegible_val.json',
                        'legible_result': 'legible_val.json',
                        'soccer_ball_list': 'soccer_ball_val.json',
                        'crops_folder': 'crops_val',
                        'crops_sr_folder': 'val/crops_sr',
                        'digit_predictions': 'val/digit_predictions.json',
                        'sim_filtered': 'val/main_subject_0.4.json',
                        'gauss_filtered': 'val/main_subject_gauss_th=3.5_r=3.json',
                        'pose_input_json': 'pose_input_val.json',
                        'pose_output_json': 'pose_results_val.json',
                        'jersey_id_result': 'jersey_id_results_validation.json',
                        'final_result': 'final_results_val.json',
                        'raw_legible_result': 'val_raw_legible_combined.json'
                    },
                 'train': {
                     'images': 'train/images',
                     'gt': 'train/train_gt.json',
                     'feature_output_folder': 'out/SoccerNetResults/train',
                     'illegible_result': 'illegible_train.json',
                     'legible_result': 'legible_train.json',
                     'soccer_ball_list': 'soccer_ball_train.json',
                     'sim_filtered': 'train/main_subject_0.4.json',
                     'gauss_filtered': 'train/main_subject_gauss_th=3.5_r=3.json',
                     'pose_input_json': 'pose_input_train.json',
                     'pose_output_json': 'pose_results_train.json',
                     'crops_folder': 'crops_train',
                     'crops_sr_folder': 'train/crops_sr',
                     'digit_predictions': 'train/digit_predictions.json',
                     'jersey_id_result': 'jersey_id_results_train.json',
                     'final_result': 'final_results_train.json',
                     'raw_legible_result': 'train_raw_legible_combined.json'
                 },
                 'challenge': {
                        'images': 'challenge/images',
                        'feature_output_folder': 'out/SoccerNetResults/challenge',
                        'gt': '',
                        'illegible_result': 'challenge_illegible.json',
                        'soccer_ball_list': 'challenge_soccer_ball.json',
                        'sim_filtered': 'challenge/main_subject_0.4.json',
                        'gauss_filtered': 'challenge/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'challenge_legible.json',
                        'pose_input_json': 'challenge_pose_input.json',
                        'pose_output_json': 'challenge_pose_results.json',
                        'crops_folder': 'challenge_crops',
                        'crops_sr_folder': 'challenge/crops_sr',
                        'digit_predictions': 'challenge/digit_predictions.json',
                        'jersey_id_result': 'challenge_jersey_id_results.json',
                        'final_result': 'challenge_final_results.json',
                        'raw_legible_result': 'challenge_raw_legible_vit.json'
                 },
                 'numbers_data': 'lmdb',

                 'legibility_model': "models/legibility_resnet34_soccer_20240215.pth",
                 'legibility_model_arch': "resnet34",

                 'legibility_model_url':  "https://drive.google.com/uc?id=1QDAqZvIbf0UPP9disdBsqcdIB0e84ZWa",
                 'pose_model_url': 'https://drive.google.com/uc?id=1gHOcfVvmwVDuJsn9c-a-v39vIqFpfbH0',
                 'str_model': 'models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt',
                 # Path to a trained TrackletAggregator checkpoint; set after training.
                 # Leave as None to use the default heuristic voting stage.
                 'aggregation_model': None,
                 'esrgan_model': 'models/RealESRGAN_x4plus.pth',

                 # ---------------------------------------------------------------------------
                 # --improved pipeline: digit classifier + LSTM aggregation
                 # ---------------------------------------------------------------------------
                 # DigitCountMoviNet (a3) checkpoint — predicts P(2-digit jersey) per tracklet.
                 # Set this to the local path after downloading from Google Drive.
                 'digit_classifier_model': 'models/digit_a3_bs16_nf256_lr4.3e-04_wd5.6e-03.ckpt',
                 'digit_classifier_model_url': 'https://drive.google.com/uc?id=11RwGs6dS60ehnIayQ0zNR6F8Am3AZXP_',
                 # TrackletAggregator (BiLSTM + digit classifier) checkpoint.
                 # Set this to the local path after downloading from Google Drive.
                 'aggregation_model_improved': 'models/agg_bs64_lr1.6e-03_wd1.1e-05_cwF_dc.ckpt',
                 'aggregation_model_improved_url': 'https://drive.google.com/uc?id=1pOdlXxRfV7cZt7oCkmQdU5zw8nV4vXAX',
                 # ---------------------------------------------------------------------------

                 #'str_model': 'pretrained=parseq',
                 'str_model_url': "https://drive.google.com/uc?id=1DULUhorGHsozOumtSocon0V-kbKwFCWG",
                },
           "Hockey": {
                 'root_dir': 'data/Hockey',
                 'legibility_data': 'legibility_dataset',
                 'numbers_data': 'jersey_number_dataset/jersey_numbers_lmdb',
                 'legibility_model':  'models/legibility_resnet34_hockey_20240201.pth',
                 'legibility_model_url':  "https://drive.google.com/uc?id=1wVmogmky9s54cn3TrO5JIcaO3R1jWpI-",
                 'str_model': 'models/parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt',
                 'str_model_url': "https://drive.google.com/uc?id=1mhGUeKUIW0-ieuCrOvfNCPgPmu1nODM7",
            }
        }