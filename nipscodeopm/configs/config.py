from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    data_path: str
    pipeline: str = "standard"
    treatment_type: str = "discrete"  # discrete, multi_binary, continuous
    treatment_col: Optional[str] = None
    treatment_cols: Optional[List[str]] = None
    outcome_col: Optional[str] = None
    outcome_sum_cols: Optional[List[str]] = None
    covariate_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None
    proxy_w_cols: Optional[List[str]] = None
    proxy_v_cols: Optional[List[str]] = None
    drop_cols: Optional[List[str]] = None
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42
    standardize: bool = True
    num_workers: int = 0
    batch_size: int = 256


@dataclass
class MaskConfig:
    use_feature_mask: bool = False
    m_conf_path: Optional[str] = None
    m_out_path: Optional[str] = None
    mask_power: float = 1.0
    mask_clip_min: float = 0.0
    mask_clip_max: float = 1.0
    kappa_phi: float = 1.0
    kappa_g: float = 1.0


@dataclass
class ModelConfig:
    latent_dim: int = 32
    nuisance_dim: int = 4
    encoder_hidden: List[int] = field(default_factory=lambda: [128, 64])
    t_encoder_hidden: List[int] = field(default_factory=lambda: [128, 64])
    t_encoder_dim: int = 32
    outcome_hidden: List[int] = field(default_factory=lambda: [128, 64])
    m_hat_hidden: List[int] = field(default_factory=lambda: [128, 64])
    pi_hidden: List[int] = field(default_factory=lambda: [128, 64])
    tau_hidden: List[int] = field(default_factory=lambda: [128, 64])
    delta_hidden: List[int] = field(default_factory=lambda: [128, 64])
    denoiser_hidden: List[int] = field(default_factory=lambda: [128, 64])
    proxy_hidden: List[int] = field(default_factory=lambda: [64, 32])
    bridge_hidden: List[int] = field(default_factory=lambda: [64, 32])
    proxy_dim: int = 8
    tau_input: str = "Z"  # X or Z
    use_projector: bool = True
    treatment_embed_dim: Optional[int] = None
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "linear"
    ddim_steps: int = 50
    self_prob: float = 0.3
    eps: float = 1e-8


@dataclass
class SSLConfig:
    enabled: bool = True
    temperature: float = 0.2
    lambda_pres: float = 1.0
    lambda_div: float = 0.1
    noise_std: float = 0.05
    drop_prob: float = 0.1
    eps: float = 1e-6


@dataclass
class LossConfig:
    lambda_self: float = 0.5
    lambda_cf: float = 1.0
    lambda_perp: float = 0.01
    lambda_Y: float = 1.0
    lambda_T: float = 1.0
    rho: float = 1e-4


@dataclass
class BasisConfig:
    basis_type: str = "poly"  # poly or spline
    degree: int = 2
    spline_knots: int = 3
    max_basis_features: int = 8


@dataclass
class DirichletConfig:
    beta_low: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.1])
    beta_high: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.2])
    kappa_min: float = 0.5
    kappa_max: float = 5.0
    ema_decay: float = 0.98
    include_ssl: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mixed_precision: bool = False
    log_every: int = 50
    eval_every: int = 1
    patience: int = 20
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda"


@dataclass
class EvalConfig:
    knn_k: int = 5
    max_eval_samples: int = 2000
    contrast_treatments: Optional[List[int]] = None
    continuous_contrast: Optional[List[float]] = None


@dataclass
class AuxLossConfig:
    use_y_aux: bool = True
    use_t_aux: bool = False


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    dirichlet: DirichletConfig = field(default_factory=DirichletConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    aux: AuxLossConfig = field(default_factory=AuxLossConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
