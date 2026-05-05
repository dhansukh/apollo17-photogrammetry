import os, glob, json
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Check both train and test renders
for split in ["train", "test"]:
    gt_dir = f"/scratch/djaladi/apollo17_output/{split}/ours_30000/gt"
    render_dir = f"/scratch/djaladi/apollo17_output/{split}/ours_30000/renders"
    if not os.path.exists(gt_dir):
        continue
    print(f"\n=== {split.upper()} SET ===")
    results = []
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        name = os.path.basename(gt_path)
        rend_path = os.path.join(render_dir, name)
        if not os.path.exists(rend_path):
            continue
        gt_img = imread(gt_path)[:, :, :3]
        rend_img = imread(rend_path)[:, :, :3]
        h = min(gt_img.shape[0], rend_img.shape[0])
        w = min(gt_img.shape[1], rend_img.shape[1])
        gt_img, rend_img = gt_img[:h, :w], rend_img[:h, :w]
        p = psnr(gt_img, rend_img)
        s = ssim(gt_img, rend_img, channel_axis=2)
        results.append({"image": name, "PSNR": round(p, 2), "SSIM": round(s, 4)})
        print(f"  {name}: PSNR={p:.2f}, SSIM={s:.4f}")
    if results:
        avg_p = np.mean([r["PSNR"] for r in results])
        avg_s = np.mean([r["SSIM"] for r in results])
        print(f"  Average: PSNR={avg_p:.2f}, SSIM={avg_s:.4f}")
