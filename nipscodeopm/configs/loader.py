from __future__ import annotations

from typing import Any, Dict

import yaml

from .config import Config, DataConfig, ModelConfig, DiffusionConfig, LossConfig, BasisConfig, DirichletConfig, TrainingConfig, EvalConfig, MaskConfig, AuxLossConfig, SSLConfig


def _update_dataclass(cls, data: Dict[str, Any]):
    return cls(**data)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    diffusion_raw = dict(raw.get("diffusion", {}))
    if "eps" in diffusion_raw:
        diffusion_raw["eps"] = float(diffusion_raw["eps"])
    data_cfg = _update_dataclass(DataConfig, raw["data"])
    model_cfg = _update_dataclass(ModelConfig, raw.get("model", {}))
    diffusion_cfg = _update_dataclass(DiffusionConfig, diffusion_raw)
    loss_cfg = _update_dataclass(LossConfig, raw.get("loss", {}))
    basis_cfg = _update_dataclass(BasisConfig, raw.get("basis", {}))
    dir_cfg = _update_dataclass(DirichletConfig, raw.get("dirichlet", {}))
    train_cfg = _update_dataclass(TrainingConfig, raw.get("training", {}))
    eval_cfg = _update_dataclass(EvalConfig, raw.get("eval", {}))
    mask_cfg = _update_dataclass(MaskConfig, raw.get("mask", {}))
    aux_cfg = _update_dataclass(AuxLossConfig, raw.get("aux", {}))
    ssl_cfg = _update_dataclass(SSLConfig, raw.get("ssl", {}))
    return Config(
        data=data_cfg,
        model=model_cfg,
        diffusion=diffusion_cfg,
        loss=loss_cfg,
        basis=basis_cfg,
        dirichlet=dir_cfg,
        training=train_cfg,
        eval=eval_cfg,
        mask=mask_cfg,
        aux=aux_cfg,
        ssl=ssl_cfg,
    )
