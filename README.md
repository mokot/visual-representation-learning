# Visual Representation Learning: Autoencoding Gaussian Splats

This repository explores **Autoencoding Gaussian Splats** for 2D image representation. Gaussian splatting has emerged as a powerful technique for modeling images and 3D scenes using parameterized Gaussians. In this project, we apply **autoencoders** to learn compact representations of images modeled as 2D Gaussian splats.

## Project Goals

1. **Construct a dataset of trained Gaussian splats** using CIFAR-10.
2. **Design and train an autoencoder** to encode and reconstruct Gaussian splat representations.
3. **Compare Gaussian splat autoencoding** with traditional pixel-based autoencoding.

## Methodology

- We use `gsplat` to fit Gaussians to CIFAR-10 images, storing parameters like position, scale, rotation, opacity, and color.
- Different **autoencoder architectures** (deep, convolutional, ResNet-based) are explored for encoding Gaussian splats.
- Experiments analyze **compression efficiency, reconstruction quality, and feature disentanglement**.

## Project Structure

```
.
├── configs/              # Configuration files for experiments
├── constants/            # Constants and transformation utilities
├── data/                 # Dataset scripts and preprocessed data
├── images/               # Visualization results (e.g., loss curves, comparisons)
├── logs/                 # Logs from different model training runs
├── models/               # Implementations of autoencoders and trainers
├── references/           # Reference implementations and utilities
├── report/               # LaTeX report files and compiled PDF
├── results/              # Experimental results and analysis
├── slurm/                # SLURM batch scripts for job scheduling
├── style/                # Custom matplotlib styles
├── submodules/           # External repositories (e.g., ResNet-18 autoencoder)
├── tests/                # Jupyter notebooks for experimenting with different setups
├── utils/                # Utility functions for data processing and visualization
├── example.ipynb         # Provided example from mentors
├── LICENSE               # License file
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mokot/visual-representation-learning.git
   cd visual-representation-learning
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run experiments:** Refer to `tests/` folder for Jupyter notebooks with different experimental setups.

## References

- **gsplat:** Open-source library for Gaussian splatting ([github.com](https://github.com/nerfstudio-project/gsplat))
- **CIFAR-10 Dataset:** Standard benchmark dataset for image representation.

## License

This project is licensed under the **MIT License**.

## Authors

Both authors contributed equally to the conceptualization, research, and implementation of this project.

- **Rok Mokotar** (LMU Munich) – [Rok.Mokotar@campus.lmu.de](mailto:Rok.Mokotar@campus.lmu.de)
- **Federico Bernardo Harjes Ruiloba** (LMU Munich) – [f.harjes@campus.lmu.de](mailto:f.harjes@campus.lmu.de)
