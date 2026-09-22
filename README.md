
# EndoGS: Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting

This is the official code for https://arxiv.org/abs/2401.11535.

## Overview

<img src='./misc/overview.jpg' width=800>

## Installation

Clone this repository and install packages:
```
git clone https://github.com/sweetpotato250/FreqGuard-LL.git
conda env create --file environment.yml
conda activate gs
pip install git+https://github.com/ingra14m/depth-diff-gaussian-rasterization.git@depth
pip install git+https://github.com/facebookresearch/pytorch3d.git
```
Note: for the submodule diff-gaussian-rasterization of the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), we use the depth branch of https://github.com/ingra14m/depth-diff-gaussian-rasterization.

## Dataset

We use the dataset in [EndoNeRF](https://github.com/med-air/EndoNeRF). Download the data from their website.

Use [COLMAP](https://demuc.de/colmap/) to estimate the initial point clouds. Store the files (`cameras.bin, images.bin, points3D.bin`) in the data path (e.g., `./data/cutting_tissues_twice/sparse/`).

## Training

python pretrain_inn.py data/cutting_tissues_twice/ --workspace output/inn_pretrain



```
python train.py {data path} --workspace {workspace}
## e.g.,
python train.py data/endonerf/cutting/ --workspace output/cutting/

python train_watermark.py data/endonerf/pulling \
    --pretrained_model_path output/clean_model \
    --workspace output/watermarked_model \
    --watermark_iters 5000 \
    --wm_len 64

```

## Inference
```
python inference.py {data path} --model_path {model path}
## e.g.,
python inference.py data/cutting_tissues_twice/ --model_path output/cutting/point_cloud/iteration_60000
```

## Evaluation
```
python eval_rgb.py --gt_dir {gt_dir path} --mask_dir {mask_dir path} --img_dir {rendered image path}
## e.g.,
python eval_rgb.py --gt_dir data/cutting_tissues_twice/images --mask_dir data/cutting_tissues_twice/gt_masks --img_dir output/cutting/point_cloud/iteration_60000/render
```
Note: we should use the same masks in training and evaluation. If the name 'gt_masks' exist, we use 'gt_masks'; if not, use 'masks'. And we exclude the unseen pixels in gt and rendered images for PSNR.

## Test
```
python test_watermark.py data/endonerf/pulling \
    --model_path output/watermarked_model/point_cloud/iteration_5000 \
    --output_path output/test_results \
    --wm_len 64
```

## Citation

If you find our work useful, please kindly cite as (v1 version of arxiv bib to avoid tracking missing):
```
@article{zhu2024deformable,
  title={Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting},
  author={Zhu, Lingting and Wang, Zhao and Jin, Zhenchao and Lin, Guying and Yu, Lequan},
  journal={arXiv preprint arXiv:2401.11535},
  year={2024}
}
```

## Acknowledgement
* The codebase is developed based on [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al.), [4D-GS](https://github.com/hustvl/4DGaussians) (Wu et al.), [SuGaR](https://github.com/Anttwo/SuGaR) (Guédon et al.), and [EndoNeRF](https://github.com/med-air/EndoNeRF) (Wang et al.).
=======
# EndoRDH

EndoRDH is a research codebase for reversible watermarking of dynamic endoscopic 3D Gaussian Splatting (3DGS) models. It embeds ownership information into spherical-harmonic (SH) carriers that are safe to edit, while keeping a key-conditioned inverse path for recovering the original host.

The design targets endoscopic scenes where some regions are clinically sensitive and the reliability of SH carriers changes across the scene because illumination is coupled to the camera. Instead of treating all Gaussian parameters equally, EndoRDH writes the watermark into perceptually low-cost, locally stable carriers.

The pipeline has four main parts:

- **Clinically admissible carrier projection.** Watermark updates are restricted to a prescribed editable SH subspace. Structural Gaussian parameters are frozen, and protected regions are excluded.
- **Riemannian photometric decoupling.** Local photometric variation is converted into a carrier-dependent transport geometry. This step does not assume recovery of the underlying illumination physics.
- **Optimal-transport-inspired reversible flow.** The watermark payload is redistributed toward low-cost, locally stable carriers, while a key-conditioned inverse recovers the original host.
- **Dual-branch redundant coding.** Error-corrected evidence is spread across complementary carrier groups, improving robustness without increasing the per-carrier perturbation budget.

Experiments on dynamic endoscopic 3DGS scenes show that EndoRDH balances representation fidelity, copyright verification, and authorized reversibility, while remaining robust to both rendering-domain and model-domain perturbations.

> This repository does not include clinical data, patient-identifiable information, private keys, or third-party model weights unless explicitly stated. Please follow the licenses and ethics requirements of the original datasets and models.
>>>>>>> d4f22fa3791a81235d9da0eb8f03cb2f9409b27f
