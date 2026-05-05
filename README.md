# Apollo 17 Photogrammetry & Gaussian Splatting

Photogrammetric reconstruction and 3D Gaussian Splatting on 15 Apollo 17 lunar surface images, processed end-to-end on the ASU Sol HPC cluster.

## Demo Videos

- [Demo Video 1](https://youtu.be/7ajr8qmjcPs)
- [Demo Video 2](https://youtu.be/xAKiqNm51yc)


## Overview

This project implements Part A of the photogrammetry assignment:
1. Process Apollo 17 imagery through a full Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline using COLMAP, producing a sparse reconstruction, dense point cloud, and textured mesh.
2. Train a 3D Gaussian Splatting model on the reconstructed scene and quantitatively compare recovered views against the originals using PSNR and SSIM.

## Dataset

15 high-resolution Apollo 17 surface photographs (`AS17-137-*` and `AS17-138-*` series) at two resolutions: 2340x2345 (8 images) and 2340x2364 (7 images).

## Results Summary

| Metric | Train (11 images) | Test (2 held-out) |
|--------|-------------------|-------------------|
| Avg PSNR | 40.92 dB | 13.38 dB |
| Avg SSIM | 0.9603 | 0.4344 |

**COLMAP sparse reconstruction:** 13/15 images registered, 17,346 points, 0.31 px mean reprojection error.

The high train PSNR confirms Gaussian Splatting faithfully recovers the original views. The lower test PSNR is expected — with only 13 registered images of repetitive lunar regolith and wide baselines between held-out views, novel-view synthesis is challenging.

## Repository Structure
.
├── compute_metrics.py        # scikit-image PSNR/SSIM evaluation script
├── run_dense_gsplat.sh       # SLURM batch script for the GPU portion of the pipeline
├── models/
│   ├── fused.ply             # Dense point cloud from COLMAP (20 MB)
│   └── meshed-poisson.ply    # Poisson-meshed surface (85 MB)
└── results/
├── train_gt/             # Ground truth training images
├── train_renders/        # Gaussian Splatting renders of training views
├── test_gt/              # Ground truth held-out images
└── test_renders/         # Gaussian Splatting renders of held-out views
## Environment

- ASU Sol HPC cluster, `public` partition with NVIDIA A100 GPU
- Miniconda environment, Python 3.10
- COLMAP (conda-forge build)
- PyTorch 2.x with CUDA 12.1
- Inria 3D Gaussian Splatting reference implementation

## Pipeline

### 1. Environment Setup

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /scratch/djaladi/miniconda3
eval "$(/scratch/djaladi/miniconda3/bin/conda init bash)"

# Accept conda TOS and create env
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n gsplat python=3.10 -y
conda activate gsplat

# Install COLMAP
conda install -c conda-forge colmap -y
```

### 2. Project Layout

```bash
mkdir -p /scratch/djaladi/apollo17/images
cp /scratch/djaladi/PNGs/*.png /scratch/djaladi/apollo17/images/
cd /scratch/djaladi/apollo17
mkdir -p sparse dense
```

### 3. COLMAP Sparse Reconstruction (CPU, login node)

The 15 images came at two slightly different resolutions, so the `--single_camera` flag could not be used; instead COLMAP was allowed to assign a per-image camera (RADIAL model).

```bash
colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.camera_model RADIAL \
    --SiftExtraction.max_image_size 4096 \
    --SiftExtraction.max_num_features 32768

colmap exhaustive_matcher \
    --database_path database.db

colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse

colmap model_analyzer --path sparse/0
```

Outcome: **13 of 15 images registered**, 17,346 sparse points, 0.31 px reprojection error. The two unregistered images had a slightly different crop and could not be matched reliably against the rest.

### 4. Undistortion (RADIAL → PINHOLE)

3D Gaussian Splatting only accepts `PINHOLE` or `SIMPLE_PINHOLE` cameras, so the dataset must be undistorted.

```bash
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path undistorted \
    --output_type COLMAP
```

The newer COLMAP places the sparse files directly in `undistorted/sparse/`, but 3DGS expects them in `undistorted/sparse/0/`, so the files were duplicated into the `0/` subfolder for compatibility.

### 5. Dense Reconstruction (GPU)

```bash
colmap patch_match_stereo \
    --workspace_path undistorted \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path undistorted \
    --workspace_format COLMAP \
    --output_path undistorted/fused.ply

colmap poisson_mesher \
    --input_path undistorted/fused.ply \
    --output_path undistorted/meshed-poisson.ply
```

Outputs: `fused.ply` (dense point cloud) and `meshed-poisson.ply` (textured mesh).

### 6. Gaussian Splatting

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install plyfile tqdm scikit-image lpips
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
```

The `--no-build-isolation` flag was required because the CUDA extensions need to find PyTorch at build time.

```bash
python train.py \
    -s /scratch/djaladi/apollo17/undistorted \
    -m /scratch/djaladi/apollo17_output \
    --iterations 30000 \
    --eval

python render.py \
    -m /scratch/djaladi/apollo17_output \
    -s /scratch/djaladi/apollo17/undistorted

python metrics.py \
    -m /scratch/djaladi/apollo17_output
```

Training ran for 30,000 iterations (~36 minutes on an A100). The `--eval` flag uses the LLFF holdout convention (every 8th image becomes a test view), giving an 11-train / 2-test split.

### 7. Quantitative Comparison (scikit-image)

The `compute_metrics.py` script computes per-image PSNR and SSIM between rendered views and ground truth using `skimage.metrics`.

```bash
python compute_metrics.py
```

## Sol-Specific Notes

- **Login vs. compute nodes:** SfM (feature extraction, matching, mapper) is CPU-bound and was run on the login node. `patch_match_stereo` and 3DGS training require GPU and were run via an interactive `srun` session or, alternatively, an `sbatch` job.
- **Conda COLMAP lacks GPU SIFT support** — the `--SiftExtraction.use_gpu` flag is rejected. CPU SIFT is fine for 15 images.
- **CUDA module:** `module load cuda/12.9` matches the PyTorch CUDA 12.1 build.

### Example SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=apollo17
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

eval "$(/scratch/djaladi/miniconda3/bin/conda init bash 2>/dev/null)"
conda activate gsplat
module load cuda/12.9
# ... pipeline commands ...
```

## Visualization

- **Point cloud / mesh (`.ply`):** Open in [MeshLab](https://www.meshlab.net/) or drag into [3dviewer.net](https://3dviewer.net).
- **Renders vs. ground truth:** Compare `results/test_gt/` with `results/test_renders/` (and the train counterparts) side by side in any image viewer.

## Discussion

The wide gap between train and test metrics is the central finding. With only 13 registered images of a homogeneous regolith surface, the Gaussian Splatting model overfits to the training views — confirmed by ~41 dB train PSNR — but lacks enough multi-view evidence to interpolate to held-out viewpoints, where PSNR drops to ~13 dB. This is consistent with the known limitations of 3DGS in sparse-input regimes and is amplified here by the scarcity of distinctive features in lunar imagery. A denser image set or per-scene priors (e.g. depth supervision, regularization on Gaussian density) would likely close the gap considerably.

## Author

DJ Pasham (Dhansukh Jaladi)
M.S. Robotics and Autonomous Systems, Arizona State University

