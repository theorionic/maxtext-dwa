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

"""DWA training: dual-optimizer, 3-phase schedule, JIT-compiled scan."""

from __future__ import annotations
import os
import time
import abc
import pickle
import queue
import threading
import logging
from typing import Any, Dict, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
import jax.sharding as js
import flax.nnx as nnx
import optax

from maxtext.dwa.config import DWATrainConfig, DWAConfig
from maxtext.dwa.model import DWALanguageModel
from maxtext.dwa.layers import PoolParam
from maxtext.dwa.losses import utilization_loss, diversity_loss, sparsity_loss, routing_entropy_loss
from maxtext.dwa.sparse_optimizer import pool_step_dense_adam, pool_step_sparse_adam, pool_step_adafactor
from maxtext.dwa.sharding import (
    replicate, shard_batch, shard_chunk, to_bf16, setup_mesh, get_mesh,
    get_data_sharding, is_main_process,
)
from maxtext.dwa.data import StreamingDataset, DataPrefetcher, get_tokenizer


def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    B, T, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, V), targets.reshape(B * T),
    ).mean()


def make_optimizer(model: nnx.Module, cfg: DWATrainConfig) -> nnx.Optimizer:
    min_lr = cfg.lr * cfg.lr_min_ratio
    if cfg.lr_scheduler == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=cfg.lr,
            warmup_steps=min(cfg.warmup_steps, max(1, cfg.max_steps - 1)),
            decay_steps=cfg.max_steps, end_value=min_lr,
        )
    else:
        schedule = optax.warmup_exponential_decay_schedule(
            init_value=0.0, peak_value=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            transition_steps=cfg.max_steps, decay_rate=0.5,
            end_value=min_lr,
        )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


class DWAModelAdapter:
    def __init__(self, cfg: DWATrainConfig, rngs: nnx.Rngs, mesh_shape: str | None = None):
        self.cfg = cfg
        mesh = setup_mesh(mesh_shape)
        self.model = to_bf16(replicate(DWALanguageModel(cfg, rngs)))
        self.opt = replicate(make_optimizer(self.model, cfg))
        self._repl = js.NamedSharding(mesh, js.PartitionSpec())
        self.alpha_ema = jax.device_put(jnp.ones(cfg.N) / cfg.N, self._repl)

        N, D = cfg.N, cfg.D
        pool_opt_name = cfg.pool_optimizer
        if pool_opt_name == "sparse_adafactor":
            self._pool_opt = (
                jax.device_put(jnp.zeros(N, dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros(D, dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros(N, dtype=jnp.int32), self._repl),
            )
        else:
            self._pool_opt = (
                jax.device_put(jnp.zeros((N, D), dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros((N, D), dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros(N, dtype=jnp.int32), self._repl),
            )

        self._graph, self._state = nnx.split(self.model)
        self._opt_graph, self._opt_state = nnx.split(self.opt)
        dwa_cfg = cfg.to_dwa_config()
        self._train_fn = _make_train_step(self._graph, self._opt_graph, dwa_cfg, cfg)
        self._eval_fn = _make_eval_step(self._graph)
        self._pool_update_fn = _make_pool_update(self._graph, cfg)

    def _get_schedule_params(self, step: int):
        warmup = self.cfg.aux_warmup_steps
        aux = min(1.0, step / warmup) if warmup > 0 else (1.0 if step >= max(1, self.cfg.phase1_end) else 0.0)

        ls_init = self.cfg.lambda_sharp_init
        if step < self.cfg.phase1_end:
            ls = ls_init
        else:
            t2 = min(1.0, (step - self.cfg.phase1_end) / max(1, self.cfg.phase2_end - self.cfg.phase1_end))
            ls = ls_init + t2 * (self.cfg.lambda_sharp_final - ls_init)

        if self.cfg.sigma_anneal:
            t_sigma = min(1.0, step / max(1, self.cfg.sigma_anneal_warmup))
            temperature = self.cfg.sigma_anneal_start + t_sigma * (self.cfg.sigma_anneal_end - self.cfg.sigma_anneal_start)
        else:
            temperature = self.cfg.T_temperature

        return float(ls), float(aux), float(temperature)

    def train_step(self, x: jax.Array, y: jax.Array, start_step: int, dynamic_params: Dict[str, float]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        if x.ndim == 2:
            x, y = x[None], y[None]
        ls, aux, temperature = self._get_schedule_params(start_step)
        dynamic_params["lambda_sharp"] = ls
        dynamic_params["temperature"] = temperature
        dynamic_params["aux_scale"] = aux
        start_step_array = jnp.array(start_step, dtype=jnp.int32)
        dp = jax.tree.map(lambda v: jnp.array(v, dtype=jnp.float32), dynamic_params)

        (metrics, self._state, self._opt_state) = self._train_fn(
            self._state, self._opt_state, x, y, start_step_array, self.alpha_ema, dp
        )
        alpha_ema_new = metrics.pop("alpha_ema_new", self.alpha_ema)
        self.alpha_ema = jax.device_put(alpha_ema_new, self._repl)

        # Pool update (separate JIT — outside scan to avoid carry bloat)
        first_x = x[0] if x.shape[0] == 1 else x[0]
        first_y = y[0] if y.shape[0] == 1 else y[0]
        self._state, self._pool_opt = self._pool_update_fn(
            self._state, self._pool_opt, first_x, first_y,
            jnp.array(ls, dtype=jnp.float32),
            jnp.array(temperature, dtype=jnp.float32),
        )

        return metrics["total"], metrics

    def eval_step(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self._eval_fn(self._state, x, y)

    def _sync_to_model(self):
        nnx.update(self.model, self._state)
        nnx.update(self.opt, self._opt_state)

    def _sync_from_model(self):
        _, self._state = nnx.split(self.model)
        _, self._opt_state = nnx.split(self.opt)

    def save_extra_state(self) -> Dict[str, Any]:
        self._sync_to_model()
        extra = {"alpha_ema": np.array(self.alpha_ema)}
        m0, m1, m2 = self._pool_opt
        extra["pool_opt_0"] = np.array(m0)
        extra["pool_opt_1"] = np.array(m1)
        extra["pool_opt_2"] = np.array(m2)
        return extra

    def load_extra_state(self, state: Dict[str, Any]):
        if "alpha_ema" in state:
            self.alpha_ema = jax.device_put(jnp.array(state["alpha_ema"]), self._repl)
        if "pool_opt_0" in state:
            self._pool_opt = (
                jax.device_put(jnp.array(state["pool_opt_0"]), self._repl),
                jax.device_put(jnp.array(state["pool_opt_1"]), self._repl),
                jax.device_put(jnp.array(state["pool_opt_2"]), self._repl),
            )
        self._sync_from_model()


def _make_train_step(graph, opt_graph, cfg: DWAConfig, lm_cfg: DWATrainConfig):
    N = cfg.N
    ema_decay = cfg.ema_decay
    phase1_end_j = jnp.array(cfg.phase1_end, dtype=jnp.int32)
    phase2_end_j = jnp.array(cfg.phase2_end, dtype=jnp.int32)
    warmup_steps_j = jnp.array(cfg.aux_warmup_steps, dtype=jnp.int32)

    @jax.jit
    def _train_step(state, opt_state, xs, ys, start_step, alpha_ema, dynamic_params):
        def get_schedule_params(step):
            t_aux = jnp.clip(step / jnp.maximum(warmup_steps_j, jnp.array(1)), 0.0, 1.0)
            aux = t_aux
            ls_init = dynamic_params["lambda_sharp_init"]
            ls_final = dynamic_params["lambda_sharp_final"]
            is_phase2 = step >= phase1_end_j
            t2 = jnp.clip((step - phase1_end_j) / jnp.maximum(phase2_end_j - phase1_end_j, jnp.array(1)), 0.0, 1.0)
            ls = jnp.where(is_phase2, ls_init + t2 * (ls_final - ls_init), ls_init)
            temperature = dynamic_params["temperature"]
            return ls, aux, temperature

        def scan_fn(carry, inputs):
            st, o_st, alpha_ema_curr, step = carry
            x, y = inputs
            ls, aux_scale, temperature = get_schedule_params(step)

            m = nnx.merge(graph, st)
            o = nnx.merge(opt_graph, o_st)

            def loss_fn(m_inner):
                logits, alpha, keys, w_norm = m_inner(x, ls, temperature)
                ce = cross_entropy(logits, y)
                p_bu = dynamic_params["beta_util"]
                p_lu = dynamic_params["lambda_util"]
                p_ld = dynamic_params["lambda_div"]
                p_ln = dynamic_params["lambda_norm"]
                p_ls = dynamic_params["lambda_sparse"]
                p_le = dynamic_params["lambda_entropy"]
                l_u = utilization_loss(alpha_ema_curr, p_bu)
                l_d = diversity_loss(alpha, keys)
                l_s = sparsity_loss(alpha)
                l_e = routing_entropy_loss(alpha)
                aux = (p_lu * l_u + p_ld * l_d + p_ln * w_norm + p_ls * l_s + p_le * l_e)
                total = ce + aux_scale * aux
                metrics = {
                    "ce": ce, "total": total, "aux": aux,
                    "l_u": l_u, "l_d": l_d, "l_s": l_s, "l_e": l_e, "w_norm": w_norm,
                    "lambda": ls, "temperature": temperature,
                    "aux_scale": aux_scale,
                    "active_vectors": jnp.sum(alpha_ema_curr > (0.1 / N)),
                }
                return total, (metrics, alpha)

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (total, (metrics, alpha)), grads = grad_fn(m)
            o.update(m, grads)

            batch_mean = jnp.mean(alpha.reshape(-1, N), axis=0)
            alpha_ema_new = ema_decay * alpha_ema_curr + (1.0 - ema_decay) * batch_mean
            metrics["alpha_ema_new"] = alpha_ema_new

            _, new_st = nnx.split(m)
            _, new_o_st = nnx.split(o)
            return (new_st, new_o_st, alpha_ema_new, step + 1), metrics

        n_steps = xs.shape[0]
        carry = (state, opt_state, alpha_ema, start_step)
        carry, all_metrics = jax.lax.scan(scan_fn, carry, (xs, ys))
        st, o_st, final_alpha_ema, _ = carry
        avg_metrics = jax.tree.map(lambda v: jnp.mean(v, axis=0), all_metrics)
        avg_metrics["alpha_ema_new"] = final_alpha_ema
        return avg_metrics, st, o_st

    return _train_step


def _make_pool_update(graph, lm_cfg: DWATrainConfig):
    from flax.nnx.transforms.autodiff import DiffState
    N = lm_cfg.N
    pool_opt_name = lm_cfg.pool_optimizer
    pool_lr = float(lm_cfg.pool_lr)
    pool_beta1 = float(lm_cfg.pool_beta1)
    pool_beta2 = float(lm_cfg.pool_beta2)
    pool_eps = float(lm_cfg.pool_eps)
    grad_clip = float(lm_cfg.grad_clip)
    diff_state = DiffState(0, PoolParam)

    @jax.jit
    def _pool_update(state, pool_opt, x, y, lambda_sharp, temperature):
        m = nnx.merge(graph, state)

        def loss_fn(m_inner):
            logits, *_ = m_inner(x, lambda_sharp, temperature, deterministic=True)
            return cross_entropy(logits, y)

        _, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(m)
        pool_grad = grads.pool.value[...]
        pool_vals = m.pool.value[...]
        g_norm = jnp.sqrt(jnp.sum(pool_grad.astype(jnp.float32) ** 2))
        pool_grad = pool_grad * jnp.minimum(1.0, grad_clip / (g_norm + 1e-6))

        if pool_opt_name == "sparse_adafactor":
            v_r, v_c, sc = pool_opt
            pool_new, v_r_new, v_c_new, sc_new = pool_step_adafactor(
                pool_vals, v_r, v_c, sc, pool_grad,
                top_idx=jnp.arange(N), lr=pool_lr, beta2=pool_beta2, eps=pool_eps,
            )
            new_pool_opt = (v_r_new, v_c_new, sc_new)
        else:
            pool_m, pool_v, sc = pool_opt
            if pool_opt_name == "sparse_adam":
                pool_new, pool_m_new, pool_v_new, sc_new = pool_step_sparse_adam(
                    pool_vals, pool_m, pool_v, sc, pool_grad,
                    top_idx=jnp.arange(N), lr=pool_lr, beta1=pool_beta1, beta2=pool_beta2, eps=pool_eps,
                )
            else:
                pool_new, pool_m_new, pool_v_new, sc_new = pool_step_dense_adam(
                    pool_vals, pool_m, pool_v, sc, pool_grad,
                    lr=pool_lr, beta1=pool_beta1, beta2=pool_beta2, eps=pool_eps,
                )
            new_pool_opt = (pool_m_new, pool_v_new, sc_new)

        m.pool.value[...] = pool_new
        _, new_state = nnx.split(m)
        return new_state, new_pool_opt

    return _pool_update


def _make_eval_step(graph):
    @jax.jit
    def _eval_step(state, x, y):
        model = nnx.merge(graph, state)
        if x.ndim == 1:
            x = x[None]
            y = y[None]
        logits, *_ = model(x, jnp.array(10.0), jnp.array(1.0), deterministic=True)
        return cross_entropy(logits, y)
    return _eval_step


class DWATrainer:
    def __init__(self, adapter: DWAModelAdapter, cfg: DWATrainConfig,
                 tokenizer=None, ckpt_dir: str = "checkpoints/dwa", log_dir: str = "logs/dwa"):
        self.adapter = adapter
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.step = 0
        self.dataset = None
        self.is_main = is_main_process()
        if self.is_main:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = None
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                pass
        self.logger = logging.getLogger("DWA_Trainer")
        self.logger.setLevel(logging.INFO)
        if self.is_main and not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
            fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
            self.logger.addHandler(fh)

    def train(self, dataset: StreamingDataset, val_dataset: Optional[StreamingDataset] = None):
        self.dataset = dataset
        max_steps = self.cfg.max_steps
        train_steps_per_exec = self.cfg.train_steps
        prefetcher = DataPrefetcher(dataset, prefetch_size=max(train_steps_per_exec * 2, 64))

        def _get_chunks():
            xs_np, ys_np = [], []
            for bx_np, by_np in prefetcher:
                xs_np.append(bx_np)
                ys_np.append(by_np)
                if len(xs_np) == train_steps_per_exec:
                    yield shard_chunk(np.stack(xs_np), np.stack(ys_np))
                    xs_np, ys_np = [], []
            if xs_np:
                yield shard_chunk(np.stack(xs_np), np.stack(ys_np))

        dyn_params = {
            "beta_util": float(self.cfg.beta_util),
            "lambda_util": float(self.cfg.lambda_util),
            "lambda_div": float(self.cfg.lambda_div),
            "lambda_norm": float(self.cfg.lambda_norm),
            "lambda_sparse": float(self.cfg.lambda_sparse),
            "lambda_entropy": float(self.cfg.lambda_entropy),
            "lambda_sharp_init": float(self.cfg.lambda_sharp_init),
            "lambda_sharp_final": float(self.cfg.lambda_sharp_final),
            "temperature": float(self.cfg.T_temperature),
        }

        t0 = time.perf_counter()
        first_step = True
        prev_loss = None

        try:
            for chunk_idx, (x_batch, y_batch) in enumerate(_get_chunks()):
                if self.step >= max_steps:
                    break
                t0_step = time.perf_counter()
                if first_step:
                    print(f"[DWA] First chunk: x={x_batch.shape}, y={y_batch.shape}. JIT compiling...")

                loss, metrics = self.adapter.train_step(x_batch, y_batch, self.step, dyn_params)

                if first_step:
                    print(f"[DWA] First step done in {time.perf_counter() - t0_step:.1f}s (includes JIT)")
                    first_step = False

                steps_computed = x_batch.shape[0]
                old_step = self.step
                self.step += steps_computed

                do_log = (self.step // self.cfg.log_every) > (old_step // self.cfg.log_every)
                do_eval = (self.step // self.cfg.eval_every) > (old_step // self.cfg.eval_every)

                if do_log or do_eval:
                    elapsed = time.perf_counter() - t0
                    val_loss = None
                    if do_eval and val_dataset is not None:
                        val_loss = self.evaluate(val_dataset)

                    if do_log and self.is_main:
                        loss_val = float(loss)
                        tok_sec = float(self.cfg.batch_size * self.cfg.seq_len * steps_computed / max(elapsed, 1e-6))
                        arrow = "↓" if prev_loss is not None and loss_val < prev_loss else "↑" if prev_loss is not None else " "
                        active = int(metrics.get("active_vectors", 0))
                        print(f"  step {self.step:>6,}  loss={loss_val:.4f}{arrow}  "
                              f"ce={float(metrics.get('ce', 0)):.4f}  aux={float(metrics.get('aux', 0)):.4f}  "
                              f"λ={float(metrics.get('lambda', 0)):.2f}  active={active}/{self.cfg.N}  "
                              f"{elapsed:.1f}s  {tok_sec:.0f} tok/s")
                        if self.writer:
                            for k, v in metrics.items():
                                if k != "alpha_ema_new":
                                    self.writer.add_scalar(f"train/{k}", float(v), self.step)
                        prev_loss = loss_val

                    if do_eval and self.step > 0 and self.is_main:
                        self.save()
                    t0 = time.perf_counter()
        finally:
            prefetcher.stop()
            if self.writer:
                self.writer.flush()
                self.writer.close()

    def evaluate(self, val_dataset: StreamingDataset, eval_steps: int = 50) -> float:
        prefetcher = DataPrefetcher(val_dataset, prefetch_size=8)
        total_loss = 0.0
        steps = 0
        try:
            for x, y in prefetcher:
                if steps >= eval_steps:
                    break
                loss = self.adapter.eval_step(x, y)
                total_loss += float(loss)
                steps += 1
        finally:
            prefetcher.stop()
        return total_loss / max(1, steps)

    def save(self):
        if not self.is_main:
            return
        path = os.path.join(self.ckpt_dir, f"step_{self.step:06d}")
        os.makedirs(path, exist_ok=True)
        self.adapter._sync_to_model()
        extra = self.adapter.save_extra_state()
        import orbax.checkpoint as ocp
        mngr_options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        mngr = ocp.CheckpointManager(path, options=mngr_options)
        _, m_state = nnx.split(self.adapter.model)
        _, o_state = nnx.split(self.adapter.opt)
        save_args = ocp.args.StandardSave({
            "model": m_state, "optimizer": o_state,
            "step": self.step, "extra": extra,
        })
        mngr.save(self.step, args=save_args)
        mngr.wait_until_finished()
        if self.dataset:
            with open(os.path.join(path, "dataset_state.pkl"), "wb") as f:
                pickle.dump(self.dataset.state_dict(), f)

    def resume(self, dataset: StreamingDataset):
        self.dataset = dataset
        import orbax.checkpoint as ocp
        latest_dir = self.ckpt_dir
        if not os.path.exists(latest_dir):
            return
        mngr_options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        mngr = ocp.CheckpointManager(latest_dir, options=mngr_options)
        step = mngr.latest_step()
        if step is None:
            return
        _, m_state = nnx.split(self.adapter.model)
        _, o_state = nnx.split(self.adapter.opt)
        abstract = {"model": m_state, "optimizer": o_state, "step": 0, "extra": {}}
        restored = mngr.restore(step, args=ocp.args.StandardRestore(abstract))
        nnx.update(self.adapter.model, restored["model"])
        nnx.update(self.adapter.opt, restored["optimizer"])
        self.step = int(restored.get("step", step))
        if restored.get("extra"):
            self.adapter.load_extra_state(restored["extra"])
        ds_path = os.path.join(latest_dir, f"step_{self.step:06d}", "dataset_state.pkl")
        if os.path.exists(ds_path):
            with open(ds_path, "rb") as f:
                self.dataset.load_state_dict(pickle.load(f))
        print(f"[DWA] Resumed from step {self.step}")


class MaxTextDWAModelAdapter(DWAModelAdapter):
    def __init__(
        self,
        cfg: DWATrainConfig,
        rngs: nnx.Rngs,
        mt_config,
        mesh,
        mesh_shape: str | None = None,
    ):
        from maxtext.dwa.sharding import setup_from_maxtext
        mesh = setup_from_maxtext(mt_config, mesh)
        self.cfg = cfg
        self.mt_config = mt_config
        self.model = to_bf16(replicate(DWALanguageModel(cfg, rngs, mt_config=mt_config, mesh=mesh)))

        from maxtext.utils.maxtext_utils import create_learning_rate_schedule
        from maxtext.optimizers.optimizers import get_optimizer
        self.lr_schedule = create_learning_rate_schedule(mt_config)
        tx = get_optimizer(mt_config, self.lr_schedule)
        self.opt = replicate(nnx.Optimizer(self.model, tx, wrt=nnx.Param))

        self._repl = js.NamedSharding(mesh, js.PartitionSpec())
        self.alpha_ema = jax.device_put(jnp.ones(cfg.N) / cfg.N, self._repl)

        N, D = cfg.N, cfg.D
        pool_opt_name = cfg.pool_optimizer
        if pool_opt_name == "sparse_adafactor":
            self._pool_opt = (
                jax.device_put(jnp.zeros(N, dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros(D, dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros(N, dtype=jnp.int32), self._repl),
            )
        else:
            self._pool_opt = (
                jax.device_put(jnp.zeros((N, D), dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros((N, D), dtype=jnp.float32), self._repl),
                jax.device_put(jnp.zeros(N, dtype=jnp.int32), self._repl),
            )

        self._graph, self._state = nnx.split(self.model)
        self._opt_graph, self._opt_state = nnx.split(self.opt)
        dwa_cfg = cfg.to_dwa_config()
        self._train_fn = _make_train_step(self._graph, self._opt_graph, dwa_cfg, cfg)
        self._eval_fn = _make_eval_step(self._graph)
        self._pool_update_fn = _make_pool_update(self._graph, cfg)


class MaxTextDWATrainer:
    def __init__(
        self,
        adapter: MaxTextDWAModelAdapter,
        cfg: DWATrainConfig,
        mt_config,
        mesh,
        tokenizer=None,
        ckpt_dir: str = "checkpoints/dwa",
    ):
        from maxtext.common.checkpointing import create_orbax_checkpoint_manager

        self.adapter = adapter
        self.cfg = cfg
        self.mt_config = mt_config
        self.mesh = mesh
        self.tokenizer = tokenizer
        self.ckpt_dir = ckpt_dir
        self.step = 0
        self.is_main = is_main_process()

        self.checkpoint_manager = create_orbax_checkpoint_manager(
            checkpoint_dir=ckpt_dir,
            enable_checkpointing=True,
            use_async=True,
            save_interval_steps=cfg.eval_every,
        )

    def train(self, data_iterator, val_data_iterator=None):
        from maxtext.common.data_loader import DataLoader
        from maxtext.utils import max_logging

        max_steps = self.cfg.max_steps
        train_steps_per_exec = self.cfg.train_steps

        dyn_params = {
            "beta_util": float(self.cfg.beta_util),
            "lambda_util": float(self.cfg.lambda_util),
            "lambda_div": float(self.cfg.lambda_div),
            "lambda_norm": float(self.cfg.lambda_norm),
            "lambda_sparse": float(self.cfg.lambda_sparse),
            "lambda_entropy": float(self.cfg.lambda_entropy),
            "lambda_sharp_init": float(self.cfg.lambda_sharp_init),
            "lambda_sharp_final": float(self.cfg.lambda_sharp_final),
            "temperature": float(self.cfg.T_temperature),
        }

        t0 = time.perf_counter()
        first_step = True
        prev_loss = None

        while self.step < max_steps:
            xs_list, ys_list = [], []
            for _ in range(train_steps_per_exec):
                try:
                    if isinstance(data_iterator, DataLoader):
                        batch = data_iterator.load_next_batch()
                        x, y = batch["inputs"], batch["targets"]
                    else:
                        x, y = next(data_iterator)
                    xs_list.append(x)
                    ys_list.append(y)
                except StopIteration:
                    break

            if not xs_list:
                break

            x_batch = jnp.stack(xs_list)
            y_batch = jnp.stack(ys_list)

            if first_step:
                max_logging.log(f"[DWA] First chunk: x={x_batch.shape}, y={y_batch.shape}. JIT compiling...")

            loss, metrics = self.adapter.train_step(x_batch, y_batch, self.step, dyn_params)

            if first_step:
                max_logging.log(f"[DWA] First step done in {time.perf_counter() - t0:.1f}s")
                first_step = False

            steps_computed = x_batch.shape[0]
            old_step = self.step
            self.step += steps_computed

            do_log = (self.step // self.cfg.log_every) > (old_step // self.cfg.log_every)
            do_eval = (self.step // self.cfg.eval_every) > (old_step // self.cfg.eval_every)

            if do_log or do_eval:
                elapsed = time.perf_counter() - t0
                if do_log and self.is_main:
                    loss_val = float(loss)
                    max_logging.log(
                        f"step {self.step:>6,}  loss={loss_val:.4f}  "
                        f"ce={float(metrics.get('ce', 0)):.4f}  "
                        f"aux={float(metrics.get('aux', 0)):.4f}  "
                        f"λ={float(metrics.get('lambda', 0)):.2f}  "
                        f"{elapsed:.1f}s"
                    )
                    prev_loss = loss_val

                if do_eval and self.step > 0:
                    self.save()
                t0 = time.perf_counter()

    def evaluate(self, eval_iterator, eval_steps: int = 50) -> float:
        total_loss = 0.0
        steps = 0
        for _ in range(eval_steps):
            try:
                if isinstance(eval_iterator, DataLoader):
                    batch = eval_iterator.load_next_batch()
                    x, y = batch["inputs"], batch["targets"]
                else:
                    x, y = next(eval_iterator)
            except StopIteration:
                break
            loss = self.adapter.eval_step(x, y)
            total_loss += float(loss)
            steps += 1
        return total_loss / max(1, steps)

    def save(self):
        if not self.is_main:
            return
        self.adapter._sync_to_model()
        extra = self.adapter.save_extra_state()
        import orbax.checkpoint as ocp
        _, m_state = nnx.split(self.adapter.model)
        _, o_state = nnx.split(self.adapter.opt)
        save_args = ocp.args.StandardSave({
            "model": m_state, "optimizer": o_state,
            "step": self.step, "extra": extra,
        })
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save(self.step, args=save_args)
            self.checkpoint_manager.wait_until_finished()

    def resume(self):
        import orbax.checkpoint as ocp
        if self.checkpoint_manager is None:
            return
        step = self.checkpoint_manager.latest_step()
        if step is None:
            return
        _, m_state = nnx.split(self.adapter.model)
        _, o_state = nnx.split(self.adapter.opt)
        abstract = {"model": m_state, "optimizer": o_state, "step": 0, "extra": {}}
        restored = self.checkpoint_manager.restore(step, args=ocp.args.StandardRestore(abstract))
        nnx.update(self.adapter.model, restored["model"])
        nnx.update(self.adapter.opt, restored["optimizer"])
        self.step = int(restored.get("step", step))
        if restored.get("extra"):
            self.adapter.load_extra_state(restored["extra"])
        from maxtext.utils import max_logging
        max_logging.log(f"[DWA] Resumed from step {self.step}")