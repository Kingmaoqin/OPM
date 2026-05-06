# NIPSCODEOPM


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
