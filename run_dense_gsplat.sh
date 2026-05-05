#!/bin/bash
#SBATCH --job-name=apollo17
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/djaladi/apollo17_%j.out
#SBATCH --error=/scratch/djaladi/apollo17_%j.err

eval "$(/scratch/djaladi/miniconda3/bin/conda init bash 2>/dev/null)"
conda activate gsplat
module load cuda/12.9

cd /scratch/djaladi/apollo17

# --- Dense reconstruction ---
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path dense \
    --output_type COLMAP

colmap patch_match_stereo \
    --workspace_path dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path dense \
    --workspace_format COLMAP \
    --output_path dense/fused.ply

colmap poisson_mesher \
    --input_path dense/fused.ply \
    --output_path dense/meshed-poisson.ply

echo "=== COLMAP DENSE DONE ==="

# --- Build 3DGS CUDA extensions if not already built ---
cd /scratch/djaladi/gaussian-splatting
pip install submodules/diff-gaussian-rasterization 2>/dev/null
pip install submodules/simple-knn 2>/dev/null

# --- Train Gaussian Splatting ---
python train.py \
    -s /scratch/djaladi/apollo17 \
    -m /scratch/djaladi/apollo17_output \
    --iterations 30000 \
    --eval

# --- Render & Metrics ---
python render.py \
    -m /scratch/djaladi/apollo17_output \
    -s /scratch/djaladi/apollo17

python metrics.py \
    -m /scratch/djaladi/apollo17_output

echo "=== ALL DONE ==="
