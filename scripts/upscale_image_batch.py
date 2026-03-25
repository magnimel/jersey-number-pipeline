import os
import subprocess
from tqdm.auto import tqdm


def upscale_image_batch(input_root, output_root, model_name="realesrgan-x4plus", gpu_id="0"):

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
        return 0, 0

    failed = 0
    total = len(all_images)

    pbar = tqdm(all_images, desc="Upscaling", unit="img")

    for in_file in pbar:
        rel_path = os.path.relpath(in_file, input_root)
        out_file = os.path.join(output_root, rel_path)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        pbar.set_postfix(file=os.path.basename(in_file)[:15])

        cmd = [
            exe_path,
            "-i", in_file,
            "-o", out_file,
            "-n", model_name,
            "-s", "4",
            "-m", models_path,
            "-g", str(gpu_id)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode != 0 or not os.path.exists(out_file):
            failed += 1

    return failed, total