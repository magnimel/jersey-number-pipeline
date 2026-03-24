import os
import subprocess
from tqdm.auto import tqdm

def upscale_image_batch(input_root, output_root, model_name="realesrgan-x4plus"):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(base_dir, "realesrgan-ncnn-vulkan")
    models_path = os.path.normpath(os.path.join(base_dir, "../models"))
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    all_images = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_images.append(os.path.join(root, f))
    
    if not all_images:
        print(f"No images found in: {input_root}")
        return

    pbar = tqdm(all_images, desc="Upscaling", unit="img")
    
    for in_file in pbar:
        rel_path = os.path.relpath(in_file, input_root)
        out_file = os.path.join(output_root, rel_path)
        
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        pbar.set_postfix(file=os.path.basename(in_file)[:15])
        
        subprocess.run([
            exe_path, 
            "-i", in_file, 
            "-o", out_file, 
            "-n", model_name, 
            "-s", "4",
            "-m", models_path
        ], capture_output=True)

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.normpath(os.path.join(current_script_dir, "../out/SoccerNetResults/crops/imgs"))
    output_path = os.path.normpath(os.path.join(current_script_dir, "../out/SoccerNetResults/crops_sr/imgs"))

    upscale_image_batch(input_path, output_path)