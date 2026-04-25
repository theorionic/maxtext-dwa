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

"""DWA Language Model — the full model assembling transformer halves around a dynamic middle."""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import Mesh

from maxtext.dwa.config import DWATrainConfig, DWAConfig
from maxtext.dwa.layers import (
    VectorPool,
    MultiAspectRetrieval,
    DWAMiddleLayer,
    TransformerBlock,
    _precompute_rope,
    _HAS_MAXTEXT_RMSNORM,
)

try:
    from maxtext.layers.embeddings import Embed as MaxTextEmbed
    _HAS_MAXTEXT_EMBED = True
except ImportError:
    MaxTextEmbed = None
    _HAS_MAXTEXT_EMBED = False

try:
    from maxtext.layers.normalizations import RMSNorm as MaxTextRMSNorm
    _HAS_MAXTEXT_RMSNORM_MODEL = True
except ImportError:
    MaxTextRMSNorm = None
    _HAS_MAXTEXT_RMSNORM_MODEL = False

_NormLayer = MaxTextRMSNorm if (_HAS_MAXTEXT_RMSNORM or _HAS_MAXTEXT_RMSNORM_MODEL) else nnx.LayerNorm


class DWALanguageModel(nnx.Module):
    def __init__(
        self,
        cfg: DWATrainConfig,
        rngs: nnx.Rngs,
        mt_config=None,
        mesh: Mesh = None,
    ) -> None:
        self.cfg = cfg
        self._use_mt_embed = mt_config is not None and mesh is not None
        self._use_tied_head = self._use_mt_embed
        dwa = cfg.to_dwa_config()
        if "dropout" not in rngs:
            rngs = nnx.Rngs(params=rngs.params(), dropout=rngs.params())

        if self._use_mt_embed:
            self.tok_emb = MaxTextEmbed(
                cfg.vocab_size, cfg.d_model, mt_config, mesh, rngs=rngs,
            )
        else:
            self.tok_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)

        self.use_rope = cfg.use_rope
        if not cfg.use_rope:
            self.pos_emb = nnx.Param(
                nnx.initializers.normal(0.02)(rngs.params(), (cfg.seq_len, cfg.d_model))
            )

        self.blocks_A = nnx.List([
            TransformerBlock(
                cfg.d_model, cfg.n_heads, dropout_rate=cfg.dropout_rate,
                use_rope=cfg.use_rope, rope_base=cfg.rope_base,
                num_kv_heads=cfg.num_kv_heads, window_size=cfg.window_size,
                rngs=rngs,
            ) for _ in range(cfg.n_layers_A)
        ])
        self.ln_mid = _NormLayer(cfg.d_model, rngs=rngs)
        self.pool = VectorPool(dwa, rngs)
        self.retrieval = MultiAspectRetrieval(dwa, rngs)
        self.middle = DWAMiddleLayer(dwa, rngs)
        self.blocks_B = nnx.List([
            TransformerBlock(
                cfg.d_model, cfg.n_heads, dropout_rate=cfg.dropout_rate,
                use_rope=cfg.use_rope, rope_base=cfg.rope_base,
                num_kv_heads=cfg.num_kv_heads, window_size=cfg.window_size,
                rngs=rngs,
            ) for _ in range(cfg.n_layers_B)
        ])
        self.ln_f = _NormLayer(cfg.d_model, rngs=rngs)
        if not self._use_tied_head:
            self.head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)
        self.drop = nnx.Dropout(rate=cfg.dropout_rate, rngs=rngs)

    def _get_rope(self, T: int) -> Tuple[Optional[jax.Array], Optional[jax.Array]]:
        if not self.use_rope:
            return None, None
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _precompute_rope(T, head_dim, base=self.cfg.rope_base)
        return cos, sin

    def __call__(
        self,
        x: jax.Array,
        lambda_sharp: jax.Array = jnp.array(0.0),
        temperature: jax.Array = jnp.array(1.0),
        deterministic: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        B, T = x.shape
        rope_cos, rope_sin = self._get_rope(self.cfg.seq_len)

        h = self.tok_emb(x)
        if not self.use_rope:
            h = h + self.pos_emb.value[:T]
        h = self.drop(h, deterministic=deterministic)

        for block in self.blocks_A:
            h = block(h, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin)

        h = self.ln_mid(h)
        h_flat = h.reshape(B * T, self.cfg.d_model)

        pool_vecs: jax.Array = self.pool.value.value
        alpha, _scores, keys, top_idx, alpha_top = self.retrieval(
            h_flat, pool_vecs, lambda_sharp, temperature
        )
        h_flat, w_norm = self.middle(h_flat, pool_vecs, top_idx, alpha_top)

        h = h_flat.reshape(B, T, self.cfg.d_model)
        for block in self.blocks_B:
            h = block(h, deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin)

        ln_f_out = self.ln_f(h)
        if self._use_tied_head:
            logits = self.tok_emb.attend(ln_f_out)
        else:
            logits = self.head(ln_f_out)
        return logits, alpha, keys, w_norm