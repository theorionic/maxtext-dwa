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

"""HuggingFace streaming data pipeline and tokenizer wrappers."""

from __future__ import annotations

import collections
import queue
import threading
from typing import Any, Dict, Iterator, Tuple

import numpy as np


def get_tokenizer(name: str = "gpt2"):
    if name == "gpt2":
        import tiktoken

        class GPT2Tokenizer:
            def __init__(self):
                self._enc = tiktoken.get_encoding("gpt2")
                self.vocab_size = self._enc.n_vocab
                self.eos_token_id = self._enc.eot_token

            def encode(self, text: str) -> list[int]:
                return self._enc.encode(text)

            def decode(self, ids: list[int]) -> str:
                return self._enc.decode(ids)

        return GPT2Tokenizer()
    else:
        from transformers import AutoTokenizer

        class HFTokenizer:
            def __init__(self, name: str):
                self._tokenizer = AutoTokenizer.from_pretrained(name)
                self.vocab_size = self._tokenizer.vocab_size
                self.eos_token_id = self._tokenizer.eos_token_id or self._tokenizer.vocab_size - 1

            def encode(self, text: str) -> list[int]:
                return self._tokenizer.encode(text)

            def decode(self, ids: list[int]) -> str:
                return self._tokenizer.decode(ids, skip_special_tokens=True)

        return HFTokenizer(name)


class StreamingDataset:
    def __init__(
        self,
        datasets_cfg: list,
        split: str,
        tokenizer: Any,
        seq_len: int,
        batch_size: int,
        chunk_size: int = 50_000,
    ):
        self.datasets_cfg = datasets_cfg
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        from datasets import load_dataset, interleave_datasets

        dss = []
        probs = []
        for d_cfg_raw in datasets_cfg:
            d_cfg = dict(d_cfg_raw)
            ds_name = d_cfg["name"]
            weight = d_cfg.get("weight", 1.0)
            subset = d_cfg.get("subset", None)
            try:
                if subset:
                    ds = load_dataset(ds_name, subset, split=split, streaming=True)
                else:
                    ds = load_dataset(ds_name, split=split, streaming=True)
                dss.append(ds)
                probs.append(weight)
            except (ValueError, Exception) as e:
                if "split" in str(e).lower() or "Bad split" in str(e):
                    continue
                raise

        if not dss:
            raise ValueError(f"No datasets successfully instantiated for split '{split}'.")

        total_weight = sum(probs)
        probs = [p / total_weight for p in probs]

        if len(dss) > 1:
            self.ds = interleave_datasets(dss, probabilities=probs, seed=42).shuffle(
                buffer_size=chunk_size, seed=42,
            )
        else:
            self.ds = dss[0].shuffle(buffer_size=chunk_size, seed=42)

        self.iterator = iter(self.ds)
        self.tokens_buffer: collections.deque = collections.deque()

    def state_dict(self) -> Dict[str, Any]:
        return {"ds_state": self.ds.state_dict(), "tokens_buffer": list(self.tokens_buffer)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state:
            self.ds.load_state_dict(state["ds_state"])
            self.iterator = iter(self.ds)
            self.tokens_buffer = collections.deque(state.get("tokens_buffer", []))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        needed = (self.seq_len + 1) * self.batch_size
        for ex in self.iterator:
            text = ex.get("text") or ex.get("content") or ""
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            self.tokens_buffer.extend(tokens)
            while len(self.tokens_buffer) >= needed:
                bx, by = [], []
                for _ in range(self.batch_size):
                    chunk = [self.tokens_buffer.popleft() for _ in range(self.seq_len + 1)]
                    bx.append(chunk[:-1])
                    by.append(chunk[1:])
                yield np.array(bx, dtype=np.int32), np.array(by, dtype=np.int32)


class DataPrefetcher:
    def __init__(self, dataset: StreamingDataset, prefetch_size: int = 512):
        self.dataset = dataset
        self.prefetch_size = prefetch_size
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            for bx_np, by_np in self.dataset:
                if self._stop_event.is_set():
                    break
                self._queue.put((bx_np, by_np))
        except Exception:
            pass
        finally:
            self._queue.put(None)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self._thread.start()
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def get_maxtext_tokenizer(config):
    from maxtext.input_pipeline.tokenizer import build_tokenizer
    return build_tokenizer(
        config.tokenizer_path,
        config.tokenizer_type,
        config.add_bos,
        config.add_eos,
        getattr(config, "hf_access_token", ""),
    )


class MaxTextDataIterator:
    def __init__(self, config, mesh):
        from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
        from maxtext.common.data_loader import DataLoader
        self.train_iterator, self.eval_iterator = create_data_iterator(config, mesh)
        self.train_loader = DataLoader(config, mesh, self.train_iterator, None)
        self.config = config
        self.mesh = mesh

    def __iter__(self):
        while True:
            batch = self.train_loader.load_next_batch()
            yield batch["inputs"], batch["targets"]

    def get_eval_batches(self, num_steps=50):
        from maxtext.common.data_loader import DataLoader
        eval_loader = DataLoader(self.config, self.mesh, self.eval_iterator, None)
        batches = []
        for _ in range(num_steps):
            batch = eval_loader.load_next_batch()
            batches.append((batch["inputs"], batch["targets"]))
        return batches