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


def pool_step_dense_adam(pool, m, v, sc, grad, lr, beta1, beta2, eps):
    g = grad.astype(jnp.float32)
    sc_new = sc + 1
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m_new / (1 - beta1 ** sc_new[:, None])
    v_hat = v_new / (1 - beta2 ** sc_new[:, None])
    pool_new = pool.astype(jnp.float32) - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return pool_new.astype(pool.dtype), m_new, v_new, sc_new


def pool_step_sparse_adam(pool, m, v, sc, grad, top_idx, lr, beta1, beta2, eps):
    g_k = grad[top_idx].astype(jnp.float32)
    m_k = m[top_idx]
    v_k = v[top_idx]
    sc_k = sc[top_idx]

    sc_k_new = sc_k + 1
    m_k_new = beta1 * m_k + (1 - beta1) * g_k
    v_k_new = beta2 * v_k + (1 - beta2) * g_k ** 2
    m_hat = m_k_new / (1 - beta1 ** sc_k_new[:, None])
    v_hat = v_k_new / (1 - beta2 ** sc_k_new[:, None])
    delta = lr * m_hat / (jnp.sqrt(v_hat) + eps)

    pool_vals = pool.astype(jnp.float32)
    pool_new = pool_vals.at[top_idx].set(pool_vals[top_idx] - delta)
    m_new = m.at[top_idx].set(m_k_new)
    v_new = v.at[top_idx].set(v_k_new)
    sc_new = sc.at[top_idx].set(sc_k_new)
    return pool_new.astype(pool.dtype), m_new, v_new, sc_new


def pool_step_adafactor(pool, v_r, v_c, sc, grad, top_idx, lr, beta2, eps, clip_threshold=1.0):
    g_k = grad[top_idx].astype(jnp.float32)
    sc_k = sc[top_idx]
    sc_k_new = sc_k + 1

    rho = beta2 ** sc_k_new
    v_r_k = v_r[top_idx]
    g_sq = g_k ** 2

    v_r_k_new = rho * v_r_k + (1 - rho) * g_sq.mean(axis=-1)
    v_c_new = beta2 * v_c + (1 - beta2) * g_sq.mean(axis=0)

    v_approx = (v_r_k_new[:, None] * v_c_new[None, :] /
                (v_r_k_new.mean() + eps))
    update = g_k / (jnp.sqrt(v_approx) + eps)

    rms = jnp.sqrt((update ** 2).mean(axis=-1, keepdims=True))
    update = update / jnp.maximum(rms / clip_threshold, 1.0)

    pool_vals = pool.astype(jnp.float32)
    pool_new = pool_vals.at[top_idx].set(pool_vals[top_idx] - lr * update)
    v_r_new = v_r.at[top_idx].set(v_r_k_new)
    sc_new = sc.at[top_idx].set(sc_k_new)
    return pool_new.astype(pool.dtype), v_r_new, v_c_new, sc_new