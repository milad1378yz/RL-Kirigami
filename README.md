# RL-Kirigami

This repo has three main entry points:

- `data_generator/generator.py`: builds the kirigami dataset
- `fm_training.py`: flow-matching training
- `rl_training.py`: RL fine-tuning starting from a flow-matching checkpoint

## Setup

If you want the convenience CLI wrappers from `pyproject.toml`, an editable install is optional:

```bash
python -m pip install -e .
```

## Main runs

### 1. Generate the dataset

```bash
python -m data_generator.generator --config configs/data_generator.yaml
```

Default outputs:

- dataset pickle: `data_generator/kirigami_x_dataset.pkl`
- preview image: `data_generator/preview.png`
- GIFs: `data_generator/gifs/`
- Prebuilt `5000/500/500` dataset: [Google Drive](https://drive.google.com/file/d/1axPzf4ZQqxoUYIf5aEJaMD0E0eLZGRXG/view?usp=sharing)

### 2. Train the flow-matching model

```bash
python fm_training.py --config_path configs/training.yaml --resume last
```

Outputs go under:

- checkpoints: `checkpoints/<run_name>/`
- TensorBoard logs: `checkpoints/tb/`

### 3. Run RL fine-tuning

```bash
python rl_training.py --config_path configs/training.yaml --init_from last --resume last
```

`--init_from last` loads the latest flow-matching checkpoint from the base run. RL checkpoints are written to `checkpoints/<run_name>_RL/`.

## What to change for hyperparameters

### Dataset settings

Edit `configs/data_generator.yaml` for data-generation hyperparameters:

- `grid_rows`, `grid_cols`: x-matrix shape
- `img_h`, `img_w`: rendered mask resolution
- `train`, `valid`: dataset sizes
- `x_min`, `x_max`: allowed x-value range
- `sampler`: `structured` family mixture or the `uniform` iid sampler
- `seed`: dataset seed

### Shared training settings

Edit `configs/training.yaml`:

- `model_config`: model architecture and tensor sizes
- `data`: dataset and generator references
- `training`: shared training settings used by the main training runs

### RL-only hyperparameters

The `rl_training` block in `configs/training.yaml` overrides the base `training` block during RL fine-tuning.

- `rl_training`

## Notes

- If your dataset pickle is not at the default location, add `data.pickle_path` in `configs/training.yaml`.
- The training config pulls `grid_rows`, `grid_cols`, `x_min`, `x_max`, and mask size from the generator config, so keep those files consistent.
