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

"""DWA core layer implementations for MaxText (Flax NNX)."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from maxtext.dwa.config import DWAConfig
try:
    from maxtext.layers.normalizations import RMSNorm as MaxTextRMSNorm
    _HAS_MAXTEXT_RMSNORM = True
except ImportError:
    MaxTextRMSNorm = None
    _HAS_MAXTEXT_RMSNORM = False


# ---------------------------------------------------------------------------
# Pool parameter type
# ---------------------------------------------------------------------------

class PoolParam(nnx.Variable):
    """Pool vector variable — intentionally NOT nnx.Param.

    Excludes pool vectors from nnx.Optimizer(wrt=nnx.Param) so a separate
    sparse optimizer can manage them independently (dual-optimizer pattern).
    """
    pass


# ---------------------------------------------------------------------------
# Vector pool
# ---------------------------------------------------------------------------

class VectorPool(nnx.Module):

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.N = cfg.N
        self.D = cfg.D
        self.value = PoolParam(
            nnx.initializers.normal(stddev=0.01)(rngs.params(), (cfg.N, cfg.D))
        )


# ---------------------------------------------------------------------------
# Multi-aspect retrieval
# ---------------------------------------------------------------------------

class MultiAspectRetrieval(nnx.Module):

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.k_max = cfg.k_max
        self.N = cfg.N
        self.soft_top_k = cfg.soft_top_k
        self.soft_top_k_temp = cfg.soft_top_k_temp
        self.retrieval_norm = cfg.retrieval_norm
        self.retrieval_norm_scale = cfg.retrieval_norm_scale
        lecun = nnx.initializers.lecun_normal()
        self.W_Q = nnx.Param(lecun(rngs.params(), (cfg.S, cfg.d_model, cfg.d_k)))
        self.W_K = nnx.Param(lecun(rngs.params(), (cfg.S, cfg.D, cfg.d_k)))
        self.aspect_logits = nnx.Param(jnp.zeros(cfg.S))
        self.tau = nnx.Param(jnp.array(cfg.tau_init))

    def __call__(
        self,
        z: jax.Array,
        pool_vectors: jax.Array,
        lambda_sharp: jax.Array,
        temperature: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # z: [B, d_model], pool_vectors: [N, D]
        queries = jnp.einsum("bd,sdq->bsq", z, self.W_Q[...])   # [B, S, d_k]
        keys = jnp.einsum("nd,sdq->nsq", pool_vectors, self.W_K[...])  # [N, S, d_k]

        # Cosine similarity per aspect
        q_norm = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        k_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        w = jax.nn.softmax(self.aspect_logits[...])  # aspect weights [S]
        scores = jnp.einsum("bsq,nsq,s->bn", q_norm, k_norm, w)  # [B, N]

        needs_top_k = self.k_max < self.N

        # --- Soft top-k branch ---
        def _soft_top_k(scores_arg):
            top_vals, top_idx = jax.lax.top_k(scores_arg, self.k_max)
            shifted = top_vals - top_vals[:, :1]
            soft_weights = jax.nn.softmax(shifted / self.soft_top_k_temp, axis=-1)
            batch_idx = jnp.arange(scores_arg.shape[0])[:, None]
            alpha = jnp.zeros_like(scores_arg)
            alpha = alpha.at[batch_idx, top_idx].set(soft_weights)
            return alpha, top_idx, soft_weights

        # --- Hard top-k branch ---
        def _hard_top_k(scores_arg):
            _, top_idx = jax.lax.top_k(scores_arg, self.k_max)
            batch_idx = jnp.arange(scores_arg.shape[0])[:, None]
            mask = jnp.zeros(scores_arg.shape, dtype=jnp.bool_)
            mask = mask.at[batch_idx, top_idx].set(True)
            masked_scores = jnp.where(mask, scores_arg, -1e9)
            alpha = jax.nn.softmax(masked_scores / temperature, axis=-1)
            alpha_top = alpha[batch_idx, top_idx]
            return alpha, top_idx, alpha_top

        # --- No top-k (use all N) branch ---
        def _no_top_k(scores_arg):
            idx = jnp.tile(jnp.arange(self.N), (scores_arg.shape[0], 1))
            alpha = jax.nn.softmax(scores_arg / temperature, axis=-1)
            return alpha, idx, alpha

        # Select branch based on config (static predicates — no recompilation)
        if needs_top_k and self.soft_top_k:
            alpha, top_idx, alpha_top = _soft_top_k(scores)
        elif needs_top_k:
            alpha, top_idx, alpha_top = _hard_top_k(scores)
        else:
            alpha, top_idx, alpha_top = _no_top_k(scores)

        # Sigmoid gating: g = σ(λ·(s − τ))
        g = jax.nn.sigmoid(lambda_sharp * (scores - self.tau[...]))
        alpha = g * alpha
        alpha = alpha / (jnp.sum(alpha, axis=-1, keepdims=True) + 1e-8)

        # Re-select top-k after gating
        def _retop_k(alpha_arg):
            _, idx2 = jax.lax.top_k(alpha_arg, self.k_max)
            bi = jnp.arange(alpha_arg.shape[0])[:, None]
            return idx2, alpha_arg[bi, idx2]

        top_idx_final, alpha_top_final = jax.lax.cond(
            needs_top_k,
            lambda a: _retop_k(a),
            lambda a: (top_idx, alpha_top),
            alpha,
        )

        return alpha, scores, keys, top_idx_final, alpha_top_final


# ---------------------------------------------------------------------------
# DWA middle layer (factorized rank-r assembly)
# ---------------------------------------------------------------------------

class DWAMiddleLayer(nnx.Module):
    """W = W_base + Σ α_i (U_i V_i) + Σ α_i b_i, with optional retrieval norm."""

    def __init__(self, cfg: DWAConfig, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        self._u_end = cfg.d_model * cfg.r
        self._v_end = self._u_end + cfg.r * cfg.d_model
        self._b_end = self._v_end + cfg.d_model
        self.W_base = nnx.Param(
            nnx.initializers.normal(0.01)(rngs.params(), (cfg.d_model, cfg.d_model))
        )
        self.b_base = nnx.Param(jnp.zeros(cfg.d_model))
        self.gamma = nnx.Param(jnp.array(cfg.gamma_init))
        self.layer_norm = (MaxTextRMSNorm if _HAS_MAXTEXT_RMSNORM else nnx.LayerNorm)(cfg.d_model, rngs=rngs)

    def __call__(
        self,
        h_A: jax.Array,
        pool_vectors: jax.Array,
        top_idx: jax.Array,
        alpha_top: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        # h_A: [B, d_model], pool_vectors: [N, D], top_idx: [B, k], alpha_top: [B, k]
        active_vectors = pool_vectors[top_idx]  # [B, k, D]

        u_end = self._u_end
        v_end = self._v_end
        b_end = self._b_end

        # Reshape pool slices into low-rank factors
        U = active_vectors[..., :u_end].reshape(-1, self.cfg.k_max, self.cfg.d_model, self.cfg.r)
        V = active_vectors[..., u_end:v_end].reshape(-1, self.cfg.k_max, self.cfg.r, self.cfg.d_model)
        bias = active_vectors[..., v_end:b_end]  # [B, k, d_model]

        # Efficient factorised computation: h@V then @U, weighted sum with α
        # h_V[b,k,r] = Σ_a h_A[b,a] V[b,k,r,a]
        HV = jnp.einsum("ba,bkra->bkr", h_A, V)
        # h_VU[b,k,c] = Σ_r HV[b,k,r] U[b,k,c,r]
        HUV = jnp.einsum("bkr,bkcr->bkc", HV, U)
        # Δh = Σ_k α[b,k] · HUV[b,k,:]
        out_delta = jnp.einsum("bk,bkc->bc", alpha_top, HUV)
        b_delta = jnp.einsum("bk,bkc->bc", alpha_top, bias)

        # Optional retrieval normalisation
        if self.cfg.retrieval_norm:
            rnorm = (1.0 / jnp.sqrt(jnp.sum(HUV ** 2, axis=-1, keepdims=True) + 1e-6)) * self.cfg.retrieval_norm_scale
            out_delta = out_delta * jnp.mean(rnorm, axis=1)
            b_delta = b_delta * jnp.mean(rnorm, axis=1)

        # W_base h + Δ + residual with γ
        h_base = jnp.einsum("ba,ca->bc", h_A, self.W_base[...]) + self.b_base[...]
        h_transformed = h_base + out_delta + b_delta
        h_out = self.layer_norm(h_A + self.gamma[...] * h_transformed)

        w_norm = jnp.mean(jnp.sum(out_delta ** 2, axis=-1))

        return h_out, w_norm


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.concatenate([jnp.cos(freqs), jnp.cos(freqs)], axis=-1)
    sin = jnp.concatenate([jnp.sin(freqs), jnp.sin(freqs)], axis=-1)
    return cos[None, :, None, :], sin[None, :, None, :]


def _apply_rope(q, k, cos, sin):
    half = q.shape[-1] // 2
    x1, x2 = q[..., :half], q[..., half:]
    q_rot = jnp.concatenate([-x2, x1], axis=-1)
    q = q * cos + q_rot * sin
    x1k, x2k = k[..., :half], k[..., half:]
    k_rot = jnp.concatenate([-x2k, x1k], axis=-1)
    k = k * cos + k_rot * sin
    return q, k


# ---------------------------------------------------------------------------
# Causal self-attention (GQA + RoPE + sliding window)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nnx.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout_rate: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        num_kv_heads: int = 0,
        window_size: int = 0,
        rngs: nnx.Rngs = None,  # type: ignore[assignment]
    ) -> None:
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else n_heads
        self.window_size = window_size
        assert n_heads % self.num_kv_heads == 0
        self.n_rep = n_heads // self.num_kv_heads

        self.W_q = nnx.Linear(d_model, n_heads * self.d_head, use_bias=False, rngs=rngs)
        self.W_kv = nnx.Linear(d_model, 2 * self.num_kv_heads * self.d_head, use_bias=False, rngs=rngs)
        self.W_out = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.drop = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        rope_cos: Optional[jax.Array] = None,
        rope_sin: Optional[jax.Array] = None,
    ) -> jax.Array:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        num_kv = self.num_kv_heads

        q = self.W_q(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)   # [B, H, T, dh]
        kv = self.W_kv(x)
        k, v = jnp.split(kv, 2, axis=-1)
        k = k.reshape(B, T, num_kv, dh)  # [B, T, num_kv, dh] for RoPE
        v = v.reshape(B, T, num_kv, dh)

        if self.use_rope and rope_cos is not None and rope_sin is not None:
            q_for_rope = q.transpose(0, 2, 1, 3)  # [B, T, H, dh]
            q_for_rope, k = _apply_rope(q_for_rope, k, rope_cos[:, :T], rope_sin[:, :T])
            q = q_for_rope.transpose(0, 2, 1, 3)

        k = k.transpose(0, 2, 1, 3)  # [B, num_kv, T, dh]
        v = v.transpose(0, 2, 1, 3)

        if num_kv != H:
            k = jnp.repeat(k, self.n_rep, axis=1)
            v = jnp.repeat(v, self.n_rep, axis=1)

        # Scaled dot-product attention
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * (dh ** -0.5)

        # Causal + optional sliding window mask
        if self.window_size > 0:
            i = jnp.arange(T)[:, None]
            j = jnp.arange(T)[None, :]
            causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            window = (j >= i - self.window_size + 1) & (j <= i)
            mask = causal & window
        else:
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))

        scores = jnp.where(mask[None, None], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.drop(attn, deterministic=deterministic)

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.W_out(out)


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class FeedForward(nnx.Module):

    def __init__(self, d_model: int, dropout_rate: float = 0.0, rngs: nnx.Rngs = None):  # type: ignore[assignment]
        self.fc1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)
        self.drop = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        return self.drop(self.fc2(jax.nn.gelu(self.fc1(x))), deterministic=deterministic)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nnx.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_rate: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        num_kv_heads: int = 0,
        window_size: int = 0,
        rngs: nnx.Rngs = None,  # type: ignore[assignment]
    ) -> None:
        self.ln1 = (MaxTextRMSNorm if _HAS_MAXTEXT_RMSNORM else nnx.LayerNorm)(d_model, rngs=rngs)
        self.attn = CausalSelfAttention(
            d_model, n_heads,
            dropout_rate=dropout_rate,
            use_rope=use_rope,
            rope_base=rope_base,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            rngs=rngs,
        )
        self.ln2 = (MaxTextRMSNorm if _HAS_MAXTEXT_RMSNORM else nnx.LayerNorm)(d_model, rngs=rngs)
        self.ffn = FeedForward(d_model, dropout_rate=dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        rope_cos: Optional[jax.Array] = None,
        rope_sin: Optional[jax.Array] = None,
    ) -> jax.Array:
        x = x + self.attn(self.ln1(x), deterministic=deterministic, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.ffn(self.ln2(x), deterministic=deterministic)
        return x