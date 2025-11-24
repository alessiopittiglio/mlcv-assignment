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

## Setup and installation

### 1. Clone the repository

```
git clone https://github.com/alessiopittiglio/mlcv-assignment.git
cd mlcv-assignment
pip install -e .
```

### 2. Create environment and install dependencies

We recommend using a virtual environment (e.g., Conda or venv):

```
# Using Conda
conda create -n mlcv_env python=3.9
conda activate mlcv_env

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The datasets used include StreetHazards and Pascal VOC for Outlier Exposure. As Pascal VOC cannot always be downloaded directly from the server, you can download Pascal VOC from the following mirror: [Pascal VOC](http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar).

## Configuration

All model architectures, training procedures, and hyperparameters are managed through YAML configuration files located in the `configs/` directory.

## Training models

To train a model, use the unified training script `scripts/train.py` with a specific configuration file:

```
python scripts/train.py --config path/to/model_config.yaml
```
