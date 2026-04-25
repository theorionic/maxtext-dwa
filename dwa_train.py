#!/usr/bin/env python3
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

"""DWA training entry point for MaxText."""

import os
import time
import argparse
import jax
import flax.nnx as nnx
import numpy as np

from maxtext.dwa.config import DWATrainConfig
from maxtext.dwa.model import DWALanguageModel
from maxtext.dwa.trainer import DWAModelAdapter, DWATrainer
from maxtext.dwa.data import StreamingDataset, get_tokenizer
from maxtext.dwa.sharding import is_main_process


def main_standalone(args):
    cfg = DWATrainConfig.from_yaml(args.config)

    if is_main_process():
        n_dev = len(jax.devices())
        print(f"[DWA] Devices: {n_dev}x {jax.devices()[0].device_kind}")
        print(f"[DWA] Config: d_model={cfg.d_model} layers={cfg.n_layers_A}+{cfg.n_layers_B} "
              f"pool={cfg.N}x{cfg.D} batch={cfg.batch_size} seq={cfg.seq_len}")
        params = cfg.count_params()
        print(f"[DWA] Total params: {params['total']:,}  pool: {params['pool']:,} ({params['pool_ratio']*100:.1f}%)")

    print("[DWA] Loading tokenizer...")
    t0 = time.perf_counter()
    tokenizer = get_tokenizer(cfg.tokenizer_name)
    print(f"[DWA] Tokenizer loaded in {time.perf_counter() - t0:.1f}s")

    print("[DWA] Initializing model + sharding + bf16...")
    t0 = time.perf_counter()
    rngs = nnx.Rngs(params=jax.random.key(42), dropout=jax.random.key(7))
    adapter = DWAModelAdapter(cfg, rngs, mesh_shape=args.mesh)
    print(f"[DWA] Model initialized in {time.perf_counter() - t0:.1f}s")

    print("[DWA] Creating datasets...")
    t0 = time.perf_counter()
    dataset = StreamingDataset(
        datasets_cfg=cfg.datasets, split="train", tokenizer=tokenizer,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size, chunk_size=1000,
    )
    val_dataset = StreamingDataset(
        datasets_cfg=cfg.datasets, split="validation", tokenizer=tokenizer,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size, chunk_size=500,
    )
    print(f"[DWA] Datasets created in {time.perf_counter() - t0:.1f}s")

    trainer = DWATrainer(adapter, cfg, tokenizer=tokenizer, ckpt_dir=args.ckpt_dir, log_dir=args.log_dir)
    trainer.resume(dataset)
    print("[DWA] Starting training...")
    trainer.train(dataset, val_dataset=val_dataset)

    if is_main_process():
        print("[DWA] Training finished.")


def main_maxtext(args):
    from maxtext.configs import pyconfig
    from maxtext.utils import maxtext_utils
    from maxtext.utils import max_logging
    from maxtext.dwa.config import DWATrainConfig
    from maxtext.dwa.trainer import MaxTextDWAModelAdapter, MaxTextDWATrainer
    from maxtext.dwa.sharding import setup_from_maxtext

    mt_config = pyconfig.initialize([None, args.maxtext_config])
    dwa_cfg = DWATrainConfig.from_maxtext_config(mt_config)

    if jax.process_index() == 0:
        params = dwa_cfg.count_params()
        max_logging.log(
            f"[DWA] Devices: {jax.device_count()}x {jax.devices()[0].device_kind}\n"
            f"[DWA] Config: d_model={dwa_cfg.d_model} layers={dwa_cfg.n_layers_A}+{dwa_cfg.n_layers_B} "
            f"pool={dwa_cfg.N}x{dwa_cfg.D} batch={dwa_cfg.batch_size} seq={dwa_cfg.seq_len}\n"
            f"[DWA] Total params: {params['total']:,}  pool: {params['pool']:,} ({params['pool_ratio']*100:.1f}%)"
        )

    mesh = maxtext_utils.get_mesh_from_config(mt_config)

    rngs = nnx.Rngs(params=jax.random.key(42), dropout=jax.random.key(7))
    adapter = MaxTextDWAModelAdapter(dwa_cfg, rngs, mt_config, mesh, mesh_shape=args.mesh)

    from maxtext.dwa.data import MaxTextDataIterator
    data_iter = MaxTextDataIterator(mt_config, mesh)

    trainer = MaxTextDWATrainer(
        adapter, dwa_cfg, mt_config, mesh,
        ckpt_dir=args.ckpt_dir,
    )
    trainer.resume()
    max_logging.log("[DWA] Starting training with MaxText pipeline...")
    trainer.train(data_iter)


def main():
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/jax_cache"))
    parser = argparse.ArgumentParser(description="DWA Training via MaxText")
    parser.add_argument("--config", type=str, default=None, help="Path to standalone DWA YAML config")
    parser.add_argument("--maxtext-config", type=str, default=None, help="Path to MaxText pyconfig YAML")
    parser.add_argument("--mesh", type=str, default=None, help="Device mesh: 'data' or 'data,fsdp'")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/dwa")
    parser.add_argument("--log-dir", type=str, default="logs/dwa_run")
    args = parser.parse_args()

    if args.maxtext_config:
        main_maxtext(args)
    elif args.config:
        main_standalone(args)
    else:
        parser.error("Either --config (standalone) or --maxtext-config must be provided")


if __name__ == "__main__":
    main()