# RL-Kirigami

Code for [Reinforcement learning for inverse silhouette design and rapid laser cutting of compact parallelogram quad kirigami prototypes](https://doi.org/10.1016/j.matdes.2026.116545), published in *Materials & Design* (2026), article 116545. RL-Kirigami performs inverse design for compact reconfigurable parallelogram quad kirigami: an OT-CFM generator proposes ratio fields conditioned on a target deployed silhouette, and GRPO aligns the generator to non-differentiable rewards (silhouette match, feasibility, ratio-field regularity). Decoded layouts can be exported as DXF for laser cutting.

<p align="center">
  <img src="assets/pipeline.png" width="88%">
</p>

## Setup

```bash
pip install -e .   # optional: installs the CLI wrappers declared in pyproject.toml
```

The three entry points below can also be invoked directly as `python <script>.py`.

## 1. Generate the dataset

```bash
python -m data_generator.generator --config configs/data_generator.yaml
```

Outputs:

- `data_generator/kirigami_x_dataset.pkl` - dataset pickle
- `data_generator/preview.png` - sample grid
- `data_generator/gifs/` - per-sample deployment animations

A prebuilt 5000 / 500 / 500 split is available on [Google Drive](https://drive.google.com/file/d/1axPzf4ZQqxoUYIf5aEJaMD0E0eLZGRXG/view?usp=sharing). If the pickle is not at the default path, set `data.pickle_path` in `configs/training.yaml`.

## 2. Train the OT-CFM prior

```bash
python fm_training.py --config_path configs/training.yaml --resume last
```

Outputs:

- `checkpoints/<run_name>/` - checkpoints
- `checkpoints/tb/` - TensorBoard logs

Use `--resume last` to continue from the most recent checkpoint, or omit it to start fresh.

## 3. RL fine-tune with GRPO

```bash
python rl_training.py --config_path configs/training.yaml --init_from last --resume last
```

`--init_from last` loads the latest OT-CFM checkpoint as the starting policy. RL checkpoints are written to `checkpoints/<run_name>_RL/`.

## Configuration

| File / block | What it controls |
|---|---|
| `configs/data_generator.yaml` | grid size, mask resolution, split sizes, x range, sampler, seed |
| `configs/training.yaml` -> `model_config` | backbone and tensor shapes |
| `configs/training.yaml` -> `data` | dataset and generator references |
| `configs/training.yaml` -> `training` | shared training settings (optimizer, batches, epochs) |
| `configs/training.yaml` -> `rl_training` | GRPO-only overrides (group size, reward weights, temperature) |

Keep the two YAML files consistent: `training.yaml` reads `grid_rows`, `grid_cols`, `x_min`, `x_max`, and the mask size from `data_generator.yaml`.

## Cite us

If you use this code, please cite:

```bibtex
@article{yazdani2026rlkirigami,
  title={Reinforcement learning for inverse silhouette design and rapid laser cutting of compact parallelogram quad kirigami prototypes},
  author={Yazdani, Milad and Shalileh, Shahriar and Shahriari, Dena},
  journal={Materials \& Design},
  year={2026},
  articleno={116545},
  doi={10.1016/j.matdes.2026.116545},
  url={https://doi.org/10.1016/j.matdes.2026.116545}
}
```
