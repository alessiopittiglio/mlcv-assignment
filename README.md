# Machine Learning for Computer Vision - Assignment

This project addresses the task of **semantic segmentation of unexpected objects on roads**, more generally known in the literature as *open-set semantic segmentation*, *open-world semantic segmentation*, or *anomaly segmentation*.

## Project structure

```
.
├── configs
├── main.ipynb
├── mlcv_openset_segmentation
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
└── stats
```

## Execution Environment

All experiments were conducted locally on a dedicated workstation with the following specifications:

- CPU: Intel Core i5-12600KF (12th Gen)
- GPU: NVIDIA RTX 3090
- RAM: 64 GB
- OS: Linux
- Python: 3.9.21

GPU acceleration is required for training and strongly recommended for inference.

## Setup and installation

### 1. Clone the repository

```
git clone https://github.com/alessiopittiglio/mlcv-assignment.git
cd mlcv-assignment
```

### 2. Create environment and install dependencies

We recommend using a virtual environment (e.g., Conda or venv):

```
# Using Conda
conda create -n mlcv_env python=3.9
conda activate mlcv_env

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Dataset

The datasets used include StreetHazards and Pascal VOC for Outlier Exposure. As Pascal VOC cannot always be downloaded directly from the server, you can download Pascal VOC from the following mirror: [Pascal VOC](http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar).

## Configuration

All model architectures, training procedures, and hyperparameters are managed through YAML configuration files located in the `configs/` directory.

## Inference

Multiple scripts are provided depending on the model type.

### Uncertainty-based model

Run inference using the uncertainty model:

```
python scripts/test.py --config configs/config_uncertainty.yaml
```

### Metric Learning model (pre-trained)

The metric learning model is used as a pre-trained model and is NOT trained within this project.

Inference is performed using:
```
python scripts/test_metric_model.py --config configs/config_metric.yaml
```

## Training

### Train base segmentation models

To train a model, use the unified training script `scripts/train.py` with a specific configuration file:

```
python scripts/train.py --config path/to/model_config.yaml
```

### Train Residual Pattern Learning module (RPL)

The RPL module is trained on top of the base model:

```
python scripts/train_rpl.py --config configs/config_residual.yaml
```

### Model Weights

All model weights required to reproduce the results can be downloaded from [here](https://placeholder-link.com). Once downloaded, the `.ckpt` files should be placed in the `checkpoints` directory.
