import os
import configuration as cfg
import json
import urllib.request
import gdown
import argparse
import subprocess


###### Colab setup utils ##############

def run_pip_install(packages):
    """Install packages using uv"""
    os.system(f"uv pip install {packages}")


def setup_reid(root):
    repo_name = "centroids-reid"
    src_url   = "https://github.com/mikwieczorek/centroids-reid.git"
    rep_path  = "./reid"

    if not repo_name in os.listdir(rep_path):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path, repo_name)}")

        # create the models folder inside repo, weights will be added to that folder later on
        models_folder_path = os.path.join(rep_path, repo_name, "models")
        os.system(f"mkdir {models_folder_path}")

        url = "https://drive.google.com/uc?export=download&id=1w9yzdP_5oJppGIM4gs3cETyLujanoHK8&confirm=t&uuid=fed3cb8a-1fad-40bd-8922-c41ededc93ae&at=ALgDtsxiC0WTza4g47gqC5VPyWg4:1679009047787"
        save_path = os.path.join(models_folder_path, "dukemtmcreid_resnet50_256_128_epoch_120.ckpt")
        urllib.request.urlretrieve(url, save_path)

        url = "https://drive.google.com/uc?export=download&id=1ZFywKEytpyNocUQd2APh2XqTe8X0HMom&confirm=t&uuid=450bb8b7-b3d0-4465-b0c9-bb6f066b205e&at=ALgDtswylGfYgY71u8ZmWx4CfhJX:1679008688985"
        save_path = os.path.join(models_folder_path, "market1501_resnet50_256_128_epoch_120.ckpt")
        urllib.request.urlretrieve(url, save_path)

        # Skip requirements installation - centroids-reid requires torch==1.7.1+cu101 which is
        # incompatible with Python 3.12. The reid module can be used with the current PyTorch
        # version already installed, or this functionality can be skipped for Python 3.12.
        # print("Note: Skipping centroids-reid requirements (requires PyTorch 1.7.1, incompatible with Python 3.12)")

def setup_pose(root):
    repo_name = "ViTPose"
    src_url   = "https://github.com/ViTAE-Transformer/ViTPose.git"
    rep_path  = "./pose"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
       # clone source repo
        os.chdir(root)
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path,repo_name)}")

    os.chdir(root)
    cwd = os.getcwd()
    
    # Upgrade setuptools FIRST to fix Python 3.12 compatibility globally
    run_pip_install("--upgrade setuptools>=70.0.0 pip")
    
    # Install mmcv directly instead of using mim (avoids setuptools compatibility issues)
    # Using mmcv-full for GPU support
    run_pip_install("mmcv")
    
    os.chdir(os.path.join(root, rep_path, "ViTPose"))
    # Install with no-build-isolation to avoid chumpy build issues
    os.system(f"uv pip install -v -e . --no-build-isolation")
    run_pip_install("timm==0.4.9 einops")
    
    os.chdir(cwd)


# clone and install str
# download the model
def setup_str(root):
    repo_name = "parseq"
    src_url   = "https://github.com/baudm/parseq.git"
    rep_path  = "./str"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path, repo_name)}")

    os.chdir(os.path.join(rep_path, repo_name))
    
    # Install torch with CUDA support for Colab
    # Use unsafe-best-match to allow fallback to PyPI for packages not in PyTorch index
    # run_pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --index-strategy unsafe-best-match")
    
    # Upgrade setuptools first to fix pywavelets build issue with Python 3.12
    run_pip_install("--upgrade setuptools>=70.0.0")
    
    # Install other requirements
    os.system("uv pip install -r requirements/core.txt")
    os.system("uv pip install -e .[train,test]")
    
    os.chdir(root)

def download_models_common(root_dir):
    repo_name = "ViTPose"
    rep_path = "./pose"

    url = cfg.dataset['SoccerNet']['pose_model_url']
    models_folder_path = os.path.join(rep_path, repo_name, "checkpoints")
    if not os.path.exists(models_folder_path):
        os.system(f"mkdir {models_folder_path}")
    save_path = os.path.join(rep_path, "ViTPose", "checkpoints", "vitpose-h.pth")
    if not os.path.isfile(save_path):
        gdown.download(url, save_path)

def download_models(root_dir, dataset):
    # download and save fine-tuned model
    save_path = os.path.join(root_dir, cfg.dataset[dataset]['str_model'])
    if not os.path.isfile(save_path):
        source_url = cfg.dataset[dataset]['str_model_url']
        gdown.download(source_url, save_path)

    save_path = os.path.join(root_dir, cfg.dataset[dataset]['legibility_model'])
    if not os.path.isfile(save_path):
        source_url = cfg.dataset[dataset]['legibility_model_url']
        gdown.download(source_url, save_path)

def setup_sam(root_dir):
    os.chdir(root_dir)
    repo_name = 'sam2'
    src_url = 'https://github.com/davda54/sam'

    if not repo_name in os.listdir(root_dir):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(root_dir, repo_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='all', help="Options: all, SoccerNet, Hockey")

    args = parser.parse_args()

    root_dir = os.getcwd()

    # common for both datasets
    setup_sam(root_dir)
    setup_pose(root_dir)
    download_models_common(root_dir)
    setup_str(root_dir)

    #SoccerNet only
    if not args.dataset == 'Hockey':
        setup_reid(root_dir)
        download_models(root_dir, 'SoccerNet')

    if not args.dataset == 'SoccerNet':
        download_models(root_dir, 'Hockey')
