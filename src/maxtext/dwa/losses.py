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

import jax
import jax.numpy as jnp


def utilization_loss(alpha_ema: jax.Array, beta: float = 0.1) -> jax.Array:
    eps = 1e-8
    N = alpha_ema.shape[0]
    uniform = jnp.ones_like(alpha_ema) / N
    alpha_clipped = jnp.clip(alpha_ema, eps, 1.0)
    alpha_norm = alpha_clipped / (jnp.sum(alpha_clipped) + eps)
    kl = jnp.sum(alpha_norm * jnp.log(alpha_norm * N + eps))
    return jnp.maximum(kl, 0.0)


def diversity_loss(alpha: jax.Array, keys: jax.Array) -> jax.Array:
    eps = 1e-8
    N, S, d_k = keys.shape
    alpha_mean = jnp.mean(alpha, axis=0)
    keys_flat = keys.reshape(N, S * d_k)
    k_norm = keys_flat / (jnp.linalg.norm(keys_flat, axis=-1, keepdims=True) + eps)
    sim = jnp.einsum("id,jd->ij", k_norm, k_norm)
    outer = jnp.outer(alpha_mean, alpha_mean)
    off_diag = 1.0 - jnp.eye(N)
    return jnp.sum(outer * sim * off_diag) / (N * (N - 1) + eps)


def norm_loss(W_assembled: jax.Array, W_base: jax.Array) -> jax.Array:
    diff = W_assembled - W_base[None]
    return jnp.mean(jnp.sum(diff ** 2, axis=(-2, -1)))


def sparsity_loss(alpha: jax.Array) -> jax.Array:
    eps = 1e-8
    return jnp.mean(-jnp.sum(alpha * jnp.log(alpha + eps), axis=-1))


def routing_entropy_loss(alpha: jax.Array) -> jax.Array:
    eps = 1e-8
    pool_usage = jnp.mean(alpha, axis=0)
    pool_usage = pool_usage / (jnp.sum(pool_usage) + eps)
    entropy = -jnp.sum(pool_usage * jnp.log(pool_usage + eps))
    max_entropy = jnp.log(alpha.shape[-1])
    return max_entropy - entropy