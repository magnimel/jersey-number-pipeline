import os
import configuration as cfg
import json
import urllib.request
import gdown
import argparse


###### common setup utils ##############

def make_conda_env(env_name, libs=""):
    os.system(f"conda create -n {env_name} -y "+libs)

def activate_conda_env(env_name):
    os.system(f"conda activate {env_name}")

def deactivate_conda_env(env_name):
    os.system(f"conda deactivate")

def conda_pyrun(env_name, exec_file, args):
    os.system(f"conda run -n {env_name} --live-stream python3 \"{exec_file}\" '{json.dumps(dict(vars(args)))}'")


def get_conda_envs():
    stream = os.popen("conda env list")
    output = stream.read()
    a = output.split()
    # print("OUTPUT", output)
    # print("a=", a)
    # Remove markers if they exist
    for item in ["*", "#", "conda", "environments:"]:
        while item in a:
            a.remove(item)
    
    return a[::2]
###########################################


def setup_reid(root):
    env_name  = cfg.reid_env
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

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8")
        cwd = os.getcwd()
        os.chdir(os.path.join(rep_path, repo_name))
        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f"conda run --live-stream -n {env_name} pip install -r requirements.txt")

        os.chdir(cwd)

# clone and install vitpose
# download the model
def setup_pose(root):
    env_name  = cfg.pose_env
    repo_name = "ViTPose"
    src_url   = "https://github.com/ViTAE-Transformer/ViTPose.git"
    rep_path  = "./pose"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
       # clone source repo
        os.chdir(root)
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path,repo_name)}")

    os.chdir(root)
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8")

        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f"conda run --live-stream -n {env_name} pip install  mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html")

        os.chdir(os.path.join(root, rep_path, "ViTPose"))
        os.system(f"conda run --live-stream -n {env_name} pip install -v -e .")
        os.system(f"conda run --live-stream -n {env_name} pip install timm==0.4.9 einops tqdm")

    os.chdir(root)  # always reset — prevents doubled path in download_models_common


# clone and install str
# download the model
def setup_str(root):
    env_name  = cfg.str_env
    repo_name = "parseq"
    src_url   = "https://github.com/baudm/parseq.git"
    rep_path  = "./str"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path, repo_name)}")

    os.chdir(os.path.join(rep_path, repo_name))

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.9")
        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        # Strip torch/torchvision from requirements.txt so we don't download the
        # CPU build only to immediately replace it with the GPU build.
        os.system("grep -Ev '^torch(vision)?==' requirements/core.txt > /tmp/req_no_torch.txt")
        os.system(f"conda run --live-stream -n {env_name} pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        os.system(f"conda run --live-stream -n {env_name} pip install -r /tmp/req_no_torch.txt -e .[train,test]")
        os.system(f"conda run --live-stream -n {env_name} pip install 'numpy<2'")

    os.chdir(root)

def download_models_common(root_dir):
    repo_name = "ViTPose"
    rep_path = "./pose"

    url = cfg.dataset['SoccerNet']['pose_model_url']
    models_folder_path = os.path.join(rep_path, repo_name, "checkpoints")
    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path, exist_ok=True)
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
    repo_name = 'sam'
    src_url = 'https://github.com/davda54/sam'

    # Migrate old clone location if necessary
    old_path = os.path.join(root_dir, 'sam2')
    new_path = os.path.join(root_dir, repo_name)
    if os.path.isdir(old_path) and not os.path.isdir(new_path):
        os.rename(old_path, new_path)

    if repo_name not in os.listdir(root_dir):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(root_dir, repo_name)}")


def setup_main_env(root_dir):
    """Create the 'jersey' conda env and install all runtime dependencies."""
    env_name = cfg.main_env
    if env_name in get_conda_envs():
        print(f"Conda env '{env_name}' already exists, skipping creation.")
        return

    print(f"Creating conda env '{env_name}' (python=3.9)...")
    make_conda_env(env_name, libs="python=3.9")
    os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip -y")

    pkgs = [
        "torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 "
        "-f https://download.pytorch.org/whl/torch_stable.html",
        "'numpy<2.0' opencv-python==4.10.0.84",
        "'setuptools<70.0.0'",
        "realesrgan basicsr",
        "pandas==2.2.3 tqdm==4.66.5 scipy==1.13.1 SoccerNet gdown",
    ]
    for pkg in pkgs:
        os.system(f"conda run --live-stream -n {env_name} pip install {pkg}")


def copy_reid_models(root_dir):
    """Copy downloaded reid checkpoints into centroids-reid/models/."""
    import shutil
    src_dir = os.path.join(root_dir, 'models')
    dst_dir = os.path.join(root_dir, 'reid', 'centroids-reid', 'models')
    os.makedirs(dst_dir, exist_ok=True)
    for fname in ['market1501_resnet50_256_128_epoch_120.ckpt',
                  'dukemtmcreid_resnet50_256_128_epoch_120.ckpt']:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            print(f"Copied {fname} to {dst_dir}")
        elif os.path.isfile(dst):
            print(f"Reid model already present: {fname}")
        else:
            print(f"Warning: source model not found: {src}")


def setup_esrgan(root_dir):
    """Install Real-ESRGAN and download the x4plus model weights."""
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    weight_path = os.path.join(models_dir, 'RealESRGAN_x4plus.pth')
    if not os.path.isfile(weight_path):
        print("Downloading RealESRGAN_x4plus weights...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        urllib.request.urlretrieve(url, weight_path)
        print(f"Saved to {weight_path}")
    else:
        print(f"RealESRGAN weights already present at {weight_path}")

    # Install the realesrgan Python package if not already available
    env_name = cfg.main_env
    check = os.popen(f"conda run -n {env_name} python -c 'import realesrgan' 2>&1").read()
    if 'ModuleNotFoundError' in check or 'No module named' in check:
        print("Installing realesrgan and basicsr packages...")
        os.system(f"conda run --live-stream -n {env_name} pip install 'setuptools<58'")
        os.system(f"conda run --live-stream -n {env_name} pip install --no-build-isolation basicsr realesrgan")
    else:
        print("realesrgan package already installed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='all', help="Options: all, SoccerNet, Hockey")

    args = parser.parse_args()

    root_dir = os.getcwd()

    # Create main runtime env first (other setup steps depend on it)
    setup_main_env(root_dir)

    # common for both datasets
    setup_sam(root_dir)
    setup_esrgan(root_dir)
    setup_pose(root_dir)
    download_models_common(root_dir)
    setup_str(root_dir)

    # SoccerNet only
    if not args.dataset == 'Hockey':
        setup_reid(root_dir)
        download_models(root_dir, 'SoccerNet')
        # NOTE: reid model copy (./models/ -> reid/centroids-reid/models/) is done
        # at the end of downloadables.py, after the models are actually downloaded.
        os.makedirs(os.path.join(root_dir, 'out', 'SoccerNetResults'), exist_ok=True)

    if not args.dataset == 'SoccerNet':
        download_models(root_dir, 'Hockey')

    # Remove macOS metadata files that can interfere with file listings
    os.system('find . -name ".DS_Store" -type f -delete')