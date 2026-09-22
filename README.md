
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
## Acknowledgement
* The codebase is developed based on [EndoGS](https://github.com/HKU-MedAI/EndoGS)(Zhu et al.), [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al.), [4D-GS](https://github.com/hustvl/4DGaussians) (Wu et al.), [SuGaR](https://github.com/Anttwo/SuGaR) (Guédon et al.), and [EndoNeRF](https://github.com/med-air/EndoNeRF) (Wang et al.).

