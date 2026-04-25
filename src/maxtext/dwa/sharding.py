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

"""TPU mesh, FSDP sharding for pool vectors, and bf16 casting."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
import flax.nnx as nnx

_mesh: js.Mesh | None = None
_data_sharding: js.NamedSharding | None = None
_replicated: js.NamedSharding | None = None
N_DEVICES: int = 1
GLOBAL_RANK: int = 0


def initialize_cluster() -> None:
    coordinator = os.environ.get("JAX_COORDINATOR_ADDR")
    if coordinator:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=int(os.environ.get("JAX_NUM_PROCESSES", "1")),
            process_id=int(os.environ.get("JAX_PROCESS_ID", "0")),
        )


def setup_mesh(mesh_shape: str | None = None) -> js.Mesh:
    global _mesh, _data_sharding, _replicated, N_DEVICES, GLOBAL_RANK
    if _mesh is not None:
        return _mesh

    initialize_cluster()

    devices = jax.devices()
    N_DEVICES = len(devices)
    GLOBAL_RANK = int(os.environ.get("JAX_PROCESS_ID", "0"))

    if N_DEVICES <= 1:
        _mesh = js.Mesh(np.array(devices), ("data",))
        _replicated = js.NamedSharding(_mesh, js.PartitionSpec())
        _data_sharding = js.NamedSharding(_mesh, js.PartitionSpec("data", None))
        return _mesh

    if mesh_shape is None:
        mesh_shape = "data" if N_DEVICES <= 4 else "data,fsdp"

    shape_parts = mesh_shape.split(",")
    if len(shape_parts) == 1:
        axis = shape_parts[0].strip()
        _mesh = js.Mesh(np.array(devices).reshape(-1), (axis,))
        _data_sharding = js.NamedSharding(_mesh, js.PartitionSpec(axis, None))
    elif len(shape_parts) == 2:
        data_axis, fsdp_axis = (s.strip() for s in shape_parts)
        n_data, n_fsdp = _auto_factor(N_DEVICES)
        _mesh = js.Mesh(np.array(devices).reshape(n_data, n_fsdp), (data_axis, fsdp_axis))
        _data_sharding = js.NamedSharding(_mesh, js.PartitionSpec(data_axis, None))
    else:
        raise ValueError(f"Unsupported mesh_shape: {mesh_shape}")

    _replicated = js.NamedSharding(_mesh, js.PartitionSpec())
    return _mesh


def _auto_factor(n: int) -> tuple[int, int]:
    if n <= 1:
        return (1, 1)
    for fsdp in range(min(n, 8), 0, -1):
        if n % fsdp == 0:
            return (n // fsdp, fsdp)
    return (n, 1)


def get_mesh() -> js.Mesh:
    if _mesh is None:
        setup_mesh()
    return _mesh


def get_data_sharding() -> js.NamedSharding:
    if _data_sharding is None:
        setup_mesh()
    return _data_sharding


def shard_batch(x: np.ndarray, y: np.ndarray) -> tuple[jax.Array, jax.Array]:
    if _data_sharding is None:
        setup_mesh()
    return jax.device_put(x, _data_sharding), jax.device_put(y, _data_sharding)


def shard_chunk(xs: np.ndarray, ys: np.ndarray) -> tuple[jax.Array, jax.Array]:
    if _mesh is None:
        setup_mesh()
    data_axis = _mesh.axis_names[0]
    n_data = _mesh.shape[data_axis]
    batch_dim = xs.shape[1]
    if batch_dim % n_data != 0:
        raise ValueError(
            f"batch_size={batch_dim} must be divisible by the number of data-parallel devices "
            f"({n_data}). Set batch_size to a multiple of {n_data} in your config."
        )
    chunk_sharding = js.NamedSharding(_mesh, js.PartitionSpec(None, data_axis, None))
    return jax.device_put(xs, chunk_sharding), jax.device_put(ys, chunk_sharding)


def replicate(model_or_opt: nnx.Module) -> nnx.Module:
    if _replicated is None:
        setup_mesh()

    graph, state = nnx.split(model_or_opt)

    mesh = get_mesh()
    has_fsdp = "fsdp" in mesh.axis_names if mesh else False

    def shard_rule(path, val):
        path_str = "".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        if has_fsdp and "pool" in path_str and val.ndim == 2:
            return js.NamedSharding(mesh, js.PartitionSpec("fsdp", None))
        return _replicated

    sharded_state = jax.tree_util.tree_map_with_path(
        lambda p, v: jax.device_put(v, shard_rule(p, v)),
        state,
    )

    return nnx.merge(graph, sharded_state)


def to_bf16(model: nnx.Module) -> nnx.Module:
    graph, state = nnx.split(model)
    state = jax.tree_util.tree_map(
        lambda v: v.astype(jnp.bfloat16) if v.dtype == jnp.float32 else v,
        state,
    )
    return nnx.merge(graph, state)


def is_main_process() -> bool:
    return GLOBAL_RANK == 0


def setup_from_maxtext(config, mesh):
    """Initialize global sharding state from MaxText config and mesh."""
    global _mesh, _data_sharding, _replicated, N_DEVICES, GLOBAL_RANK
    _mesh = mesh
    N_DEVICES = jax.device_count()
    GLOBAL_RANK = jax.process_index()
    _replicated = js.NamedSharding(mesh, js.PartitionSpec())
    data_axis = mesh.axis_names[0] if len(mesh.axis_names) > 0 else None
    if data_axis:
        _data_sharding = js.NamedSharding(mesh, js.PartitionSpec(data_axis, None))
    else:
        _data_sharding = _replicated
    return _mesh