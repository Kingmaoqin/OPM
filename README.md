# NIPSCODEOPM

This folder contains the paper-facing implementation of the main Confound Diffusion OPM algorithm only.

Included:

- `nipscodeopm/models/`: encoder, treatment/outcome heads, bridge/proxy networks, nuisance projector, and diffusion modules.
- `nipscodeopm/losses/`: OPM moment losses, counterfactual orthogonality loss, and adaptive Dirichlet loss weighting.
- `nipscodeopm/training/`: the core trainer that combines SSL, diffusion, and OPM objectives.
- `nipscodeopm/configs/`: dataclass configuration and YAML loader.
- `nipscodeopm/factory.py`: `build_model_from_config(...)` helper for constructing the algorithm model from config.

Not included:

- Baseline implementations.
- Experiment launchers, grid searches, paper-table builders, rebuttal scripts, checkpoints, logs, or result files.
- Dataset preprocessing and evaluation pipelines.

Minimal usage:

```python
import torch
from torch.utils.data import DataLoader

from nipscodeopm import Config, DataConfig, Trainer, build_model_from_config

cfg = Config(data=DataConfig(data_path="data.csv", treatment_type="discrete"))
model = build_model_from_config(cfg, x_dim=64, t_dim=4)

train_loader = DataLoader(...)
val_loader = DataLoader(...)
trainer = Trainer(cfg, model, train_loader, val_loader, device=torch.device("cuda"))
trainer.fit()
```

Expected batch keys:

- Required: `x`, `t`, `y`
- Optional observed proxies: `w`, `v`

The model supports discrete, multi-binary, and continuous treatments through `cfg.data.treatment_type`.
