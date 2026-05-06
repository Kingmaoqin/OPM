from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .configs import Config
from .models import ConfoundDiffusionOPM, FeatureMask


def make_feature_masks(
    cfg: Config,
    confounder_mask: Optional[np.ndarray] = None,
    treatment_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[FeatureMask], Optional[FeatureMask]]:
    if not cfg.mask.use_feature_mask:
        return None, None
    mask = cfg.mask
    phi_mask = FeatureMask(confounder_mask, mask.mask_clip_min, mask.mask_clip_max, mask.mask_power)
    g_mask = FeatureMask(treatment_mask, mask.mask_clip_min, mask.mask_clip_max, mask.mask_power)
    return phi_mask, g_mask


def build_model_from_config(
    cfg: Config,
    x_dim: int,
    t_dim: int,
    confounder_mask: Optional[np.ndarray] = None,
    treatment_mask: Optional[np.ndarray] = None,
) -> ConfoundDiffusionOPM:
    model_cfg = cfg.model
    basis_cfg = cfg.basis
    diffusion_cfg = cfg.diffusion
    encoder_mask, treatment_encoder_mask = make_feature_masks(cfg, confounder_mask, treatment_mask)

    return ConfoundDiffusionOPM(
        x_dim=x_dim,
        t_dim=t_dim,
        treatment_type=cfg.data.treatment_type,
        latent_dim=model_cfg.latent_dim,
        nuisance_dim=model_cfg.nuisance_dim,
        encoder_hidden=model_cfg.encoder_hidden,
        t_encoder_hidden=model_cfg.t_encoder_hidden,
        t_encoder_dim=model_cfg.t_encoder_dim,
        outcome_hidden=model_cfg.outcome_hidden,
        m_hat_hidden=model_cfg.m_hat_hidden,
        pi_hidden=model_cfg.pi_hidden,
        tau_hidden=model_cfg.tau_hidden,
        delta_hidden=model_cfg.delta_hidden,
        denoiser_hidden=model_cfg.denoiser_hidden,
        proxy_hidden=model_cfg.proxy_hidden,
        bridge_hidden=model_cfg.bridge_hidden,
        proxy_w_dim=model_cfg.proxy_dim,
        proxy_v_dim=model_cfg.proxy_dim,
        basis_type=basis_cfg.basis_type,
        basis_degree=basis_cfg.degree,
        spline_knots=basis_cfg.spline_knots,
        max_basis_features=basis_cfg.max_basis_features,
        mask=encoder_mask,
        t_mask=treatment_encoder_mask,
        dropout=model_cfg.dropout,
        tau_input=model_cfg.tau_input,
        use_projector=model_cfg.use_projector,
        diffusion_steps=diffusion_cfg.timesteps,
        beta_schedule=diffusion_cfg.beta_schedule,
    )
