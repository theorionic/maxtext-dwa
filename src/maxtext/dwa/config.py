# Copyright 2024-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DWA (Dynamic Weight Assembly) configuration."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import yaml


@dataclass(frozen=True)
class DWAConfig:
    """Core DWA hyperparameters (pool + retrieval + middle layer)."""
    N: int = 512                # Pool size (number of vectors)
    D: int = 4096               # Vector dimension
    d_model: int = 256          # Model hidden dimension (d_A == d_B == d_model)
    r: int = 4                  # Assembly rank per vector
    S: int = 2                  # Number of retrieval aspects
    d_k: int = 64               # Retrieval key/query dimension per aspect
    k_max: int = 16             # Max vectors retrieved per token
    gamma_init: float = 0.01    # LoRA-style residual scale init
    tau_init: float = 0.0       # Sigmoid gate threshold init
    T_temperature: float = 1.0  # Retrieval softmax temperature
    # Lambda sharp schedule
    lambda_sharp_init: float = 0.1
    lambda_sharp_phase2_end: float = 5.0
    lambda_sharp_final: float = 10.0
    # Loss weights
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.01
    lambda_entropy: float = 0.01
    # Aux schedule
    aux_warmup_steps: int = 100
    # Soft top-k
    soft_top_k: bool = True
    soft_top_k_temp: float = 1.0
    # Retrieval normalization
    retrieval_norm: bool = True
    retrieval_norm_scale: float = 0.5
    # Sigma annealing
    sigma_anneal: bool = True
    sigma_anneal_start: float = 2.0
    sigma_anneal_end: float = 0.5
    sigma_anneal_warmup: int = 200
    # Misc
    beta_util: float = 0.1
    ema_decay: float = 0.99
    phase1_end: int = 1_000
    phase2_end: int = 10_000

    def __post_init__(self) -> None:
        required = self.d_model * self.r + self.r * self.d_model + self.d_model
        assert self.D >= required, f"D={self.D} < {required} required for d_model={self.d_model}, r={self.r}"


@dataclass(frozen=True)
class DWATrainConfig:
    """Full DWA training configuration."""
    # Model architecture
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_layers_A: int = 4
    n_layers_B: int = 4
    seq_len: int = 512
    dropout_rate: float = 0.1
    # RoPE
    use_rope: bool = True
    rope_base: float = 10000.0
    # GQA
    num_kv_heads: int = 0       # 0 = full MHA
    # Window attention
    window_size: int = 0        # 0 = full causal
    # Flash attention
    use_flash: bool = True      # Use jax.nn.dot_product_attention on TPU/GPU
    # DWA pool
    N: int = 256
    D: int = 4096
    r: int = 4
    S: int = 2
    d_k: int = 64
    k_max: int = 16
    gamma_init: float = 0.01
    tau_init: float = 0.0
    T_temperature: float = 1.0
    lambda_sharp_init: float = 0.1
    lambda_sharp_phase2_end: float = 5.0
    lambda_sharp_final: float = 10.0
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.01
    lambda_entropy: float = 0.01
    aux_warmup_steps: int = 100
    soft_top_k: bool = True
    soft_top_k_temp: float = 1.0
    retrieval_norm: bool = True
    retrieval_norm_scale: float = 0.5
    sigma_anneal: bool = True
    sigma_anneal_start: float = 2.0
    sigma_anneal_end: float = 0.5
    sigma_anneal_warmup: int = 200
    beta_util: float = 0.1
    ema_decay: float = 0.99
    # Training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 20_000
    lr_scheduler: str = "cosine"
    lr_min_ratio: float = 0.1
    eval_every: int = 500
    log_every: int = 100
    eval_steps: int = 50
    phase1_end: int = 1_000
    phase2_end: int = 10_000
    grad_clip: float = 1.0
    # Pool optimizer
    pool_optimizer: str = "sparse_adam"
    pool_lr: float = 3e-4
    pool_beta1: float = 0.9
    pool_beta2: float = 0.999
    pool_eps: float = 1e-8
    # Tokenizer
    tokenizer_name: str = "gpt2"
    # Generation
    generate_every: int = 500
    generate_length: int = 50
    generate_prompts: Tuple = ("Once upon a time",)
    generate_top_k: int = 50
    # Datasets
    datasets: Optional[list] = None
    # Inner loop steps per JIT dispatch
    train_steps: int = 16
    # Gradient accumulation (optimizer updates every N gradient steps)
    grad_accum_steps: int = 1

    def __post_init__(self) -> None:
        required = self.d_model * self.r + self.r * self.d_model + self.d_model
        assert self.D >= required, f"D={self.D} < {required} required."
        if self.datasets is None:
            object.__setattr__(self, 'datasets', [{"name": "roneneldan/TinyStories", "weight": 1.0}])
        if isinstance(self.generate_prompts, list):
            object.__setattr__(self, 'generate_prompts', tuple(self.generate_prompts))

    @classmethod
    def from_yaml(cls, path: str) -> "DWATrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dwa_config(self) -> DWAConfig:
        """Extract the DWA-specific subset."""
        return DWAConfig(
            N=self.N, D=self.D, d_model=self.d_model,
            r=self.r, S=self.S, d_k=self.d_k, k_max=self.k_max,
            gamma_init=self.gamma_init, tau_init=self.tau_init,
            T_temperature=self.T_temperature,
            lambda_sharp_init=self.lambda_sharp_init,
            lambda_sharp_phase2_end=self.lambda_sharp_phase2_end,
            lambda_sharp_final=self.lambda_sharp_final,
            lambda_util=self.lambda_util, lambda_div=self.lambda_div,
            lambda_norm=self.lambda_norm, lambda_sparse=self.lambda_sparse,
            lambda_entropy=self.lambda_entropy,
            aux_warmup_steps=self.aux_warmup_steps,
            soft_top_k=self.soft_top_k, soft_top_k_temp=self.soft_top_k_temp,
            retrieval_norm=self.retrieval_norm, retrieval_norm_scale=self.retrieval_norm_scale,
            sigma_anneal=self.sigma_anneal,
            sigma_anneal_start=self.sigma_anneal_start,
            sigma_anneal_end=self.sigma_anneal_end,
            sigma_anneal_warmup=self.sigma_anneal_warmup,
            beta_util=self.beta_util, ema_decay=self.ema_decay,
            phase1_end=self.phase1_end, phase2_end=self.phase2_end,
        )

    @classmethod
    def from_maxtext_config(cls, config) -> "DWATrainConfig":
        """Create a DWATrainConfig from a MaxText HyperParameters config object.

        Maps MaxText config fields to DWA fields where applicable.
        DWA-specific fields (pool, retrieval, schedule) use DWA defaults.
        """
        kwargs = {}
        # Direct mappings from MaxText config
        if hasattr(config, "vocab_size"):
            kwargs["vocab_size"] = config.vocab_size
        if hasattr(config, "base_emb_dim"):
            kwargs["d_model"] = config.base_emb_dim
        if hasattr(config, "base_num_query_heads"):
            kwargs["n_heads"] = config.base_num_query_heads
        if hasattr(config, "max_target_length"):
            kwargs["seq_len"] = config.max_target_length
        if hasattr(config, "base_num_decoder_layers"):
            n_layers = config.base_num_decoder_layers
            if hasattr(config, "dwa_n_layers_a"):
                kwargs["n_layers_A"] = config.dwa_n_layers_a
            else:
                kwargs["n_layers_A"] = max(1, n_layers // 2)
            if hasattr(config, "dwa_n_layers_b"):
                kwargs["n_layers_B"] = config.dwa_n_layers_b
            else:
                kwargs["n_layers_B"] = max(1, n_layers // 2)
        if hasattr(config, "dropout_rate"):
            kwargs["dropout_rate"] = config.dropout_rate
        
        # DWA-specific fields (optional overrides from MaxText config)
        dwa_fields = [
            "dwa_N", "dwa_D", "dwa_r", "dwa_S", "dwa_d_k", "dwa_k_max",
            "dwa_gamma_init", "dwa_tau_init", "dwa_T_temperature",
            "dwa_lambda_sharp_init", "dwa_lambda_sharp_phase2_end", "dwa_lambda_sharp_final",
            "dwa_lambda_util", "dwa_lambda_div", "dwa_lambda_norm",
            "dwa_lambda_sparse", "dwa_lambda_entropy", "dwa_aux_warmup_steps",
            "dwa_soft_top_k", "dwa_soft_top_k_temp",
            "dwa_retrieval_norm", "dwa_retrieval_norm_scale",
            "dwa_sigma_anneal", "dwa_sigma_anneal_start", "dwa_sigma_anneal_end", "dwa_sigma_anneal_warmup",
            "dwa_beta_util", "dwa_ema_decay",
            "dwa_phase1_end", "dwa_phase2_end",
        ]
        field_map = {
            "dwa_N": "N", "dwa_D": "D", "dwa_r": "r", "dwa_S": "S",
            "dwa_d_k": "d_k", "dwa_k_max": "k_max", "dwa_gamma_init": "gamma_init",
            "dwa_tau_init": "tau_init", "dwa_T_temperature": "T_temperature",
            "dwa_lambda_sharp_init": "lambda_sharp_init",
            "dwa_lambda_sharp_phase2_end": "lambda_sharp_phase2_end",
            "dwa_lambda_sharp_final": "lambda_sharp_final",
            "dwa_lambda_util": "lambda_util", "dwa_lambda_div": "lambda_div",
            "dwa_lambda_norm": "lambda_norm", "dwa_lambda_sparse": "lambda_sparse",
            "dwa_lambda_entropy": "lambda_entropy",
            "dwa_aux_warmup_steps": "aux_warmup_steps",
            "dwa_soft_top_k": "soft_top_k", "dwa_soft_top_k_temp": "soft_top_k_temp",
            "dwa_retrieval_norm": "retrieval_norm", "dwa_retrieval_norm_scale": "retrieval_norm_scale",
            "dwa_sigma_anneal": "sigma_anneal", "dwa_sigma_anneal_start": "sigma_anneal_start",
            "dwa_sigma_anneal_end": "sigma_anneal_end", "dwa_sigma_anneal_warmup": "sigma_anneal_warmup",
            "dwa_beta_util": "beta_util", "dwa_ema_decay": "ema_decay",
            "dwa_phase1_end": "phase1_end", "dwa_phase2_end": "phase2_end",
        }
        for mt_field, dwa_field in field_map.items():
            if hasattr(config, mt_field):
                kwargs[dwa_field] = getattr(config, mt_field)

        # Training hyperparameters from MaxText config
        if hasattr(config, "global_batch_size_to_train_on"):
            kwargs["batch_size"] = config.global_batch_size_to_train_on
        if hasattr(config, "learning_rate"):
            kwargs["lr"] = config.learning_rate
        if hasattr(config, "adam_weight_decay"):
            kwargs["weight_decay"] = config.adam_weight_decay
        if hasattr(config, "learning_rate_schedule"):
            kwargs["lr_scheduler"] = config.learning_rate_schedule
        if hasattr(config, "gradient_clipping_threshold"):
            kwargs["grad_clip"] = config.gradient_clipping_threshold
        if hasattr(config, "steps"):
            kwargs["max_steps"] = config.steps
        if hasattr(config, "gradient_accumulation_steps"):
            kwargs["grad_accum_steps"] = config.gradient_accumulation_steps

        # GQA / attention config
        if hasattr(config, "base_num_kv_heads"):
            kwargs["num_kv_heads"] = config.base_num_kv_heads
        if hasattr(config, "sliding_window_size") and hasattr(config, "attention_type"):
            if getattr(config, "attention_type", None) and config.attention_type.value == "local_sliding":
                kwargs["window_size"] = config.sliding_window_size

        return cls(**kwargs)

    def count_params(self) -> dict:
        """Count parameters for every component."""
        d = self.d_model
        V = self.vocab_size
        h = self.n_heads
        dh = d // h
        kv = self.num_kv_heads if self.num_kv_heads > 0 else h
        ffn = 4 * d

        emb = V * d
        head = d * V
        pos = 0 if self.use_rope else self.seq_len * d

        attn_q = d * (h * dh)
        attn_kv = d * (2 * kv * dh)
        attn_out = d * d
        ln1 = 2 * d
        fc1 = d * ffn
        fc2 = ffn * d
        ln2 = 2 * d
        per_block = attn_q + attn_kv + attn_out + ln1 + fc1 + fc2 + ln2

        blocks_a = self.n_layers_A * per_block
        blocks_b = self.n_layers_B * per_block
        pool = self.N * self.D
        ret_wq = self.S * d * self.d_k
        ret_wk = self.S * self.D * self.d_k
        ret_logits = self.S
        ret_tau = 1
        retrieval = ret_wq + ret_wk + ret_logits + ret_tau
        mid_w = d * d
        mid_b = d
        mid_gamma = 1
        mid_ln = 2 * d
        middle = mid_w + mid_b + mid_gamma + mid_ln
        ln_mid = 2 * d
        ln_f = 2 * d

        part_a = emb + pos + blocks_a + retrieval + middle + ln_mid
        part_b = blocks_b + ln_f + head
        total = part_a + pool + part_b

        return {
            "embedding": emb + pos,
            "blocks_a": blocks_a,
            "blocks_b": blocks_b,
            "pool": pool,
            "retrieval": retrieval,
            "middle": middle,
            "head": head,
            "total": total,
            "pool_ratio": pool / total,
        }