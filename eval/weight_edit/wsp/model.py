"""Run Qwen3-Embedding: last-token pooling over left-padded batches, L2-normalized (fp16 on MPS, fp32 on CPU).

`Encoder.encode` returns embeddings, `Encoder.acts` returns the last-token residual stream at chosen depths (the "read"
pass used by editing agents). Depth k means "after k transformer blocks": 0 = embedding output, 28 = last block output
before the final norm.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from . import util


def dtype_name(device: str) -> str:
    return "fp16" if device == "mps" else "fp32"


class Encoder:
    def __init__(self, model_dir: Path, device: str = "cpu", batch_tokens: int | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if device == "mps" else torch.float32
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("WSP_THREADS", "6")))
            util.lower_priority()
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.tok.padding_side = "left"
        self.model = AutoModel.from_pretrained(str(model_dir), dtype=self.dtype).to(device).eval()
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.batch_tokens = batch_tokens or (32768 if device == "mps" else 4096)
        self.model_dir = Path(model_dir)
        self.single_fallback = device != "cpu"  # redo a NaN batch without padding (GPU backends only)
        self.n_fallback_batches = 0

    # --- tokenization and batching ---

    def tokenize(self, texts: list[str], max_len: int) -> list[list[int]]:
        return self.tok(texts, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]

    def _batches(self, lengths: np.ndarray) -> list[np.ndarray]:
        """Index batches over texts sorted by length (longest first), each at most batch_tokens padded tokens."""
        order = np.argsort(-lengths, kind="stable")
        batches: list[np.ndarray] = []
        start = 0
        while start < len(order):
            longest = int(lengths[order[start]])
            n = max(1, self.batch_tokens // max(longest, 1))
            batches.append(order[start : start + n])
            start += n
        return batches

    def _tensors(self, ids: list[list[int]], idx: np.ndarray):
        torch = self.torch
        width = max(len(ids[i]) for i in idx)
        input_ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            row = ids[i]
            input_ids[r, width - len(row) :] = torch.tensor(row)
            mask[r, width - len(row) :] = 1
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        return input_ids.to(self.device), mask.to(self.device), position_ids.to(self.device)

    # --- embeddings ---

    def encode(self, texts: list[str], max_len: int, resume_path: Path | None = None, label: str = "") -> np.ndarray:
        torch = self.torch
        n = len(texts)
        out = np.zeros((n, self.model.config.hidden_size), dtype=np.float32)
        if n == 0:
            return out
        ids = self.tokenize(texts, max_len)
        lengths = np.array([len(x) for x in ids])
        batches = self._batches(lengths)
        text_hash = util.sha256_texts(texts)
        done = np.zeros(n, dtype=bool)
        if resume_path is not None and resume_path.exists():
            z = np.load(resume_path, allow_pickle=False)
            if str(z["text_hash"]) == text_hash and str(z["device"]) == self.device:
                out, done = z["emb"], z["done"]
                print(f"  resuming {label}: {int(done.sum())}/{n} texts already encoded", file=sys.stderr)
        total_tokens = int(lengths.sum())
        t0 = time.time()
        last_log = last_save = t0
        tokens_done = int(lengths[done].sum())
        with torch.inference_mode():
            for b in batches:
                if done[b].all():
                    continue
                input_ids, mask, pos = self._tensors(ids, b)
                h = self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1]
                h = h.float()
                if not bool(torch.isfinite(h).all()):
                    h = self._retry_unpadded(ids, b)
                h = torch.nn.functional.normalize(h, p=2, dim=1)
                out[b] = h.cpu().numpy()
                done[b] = True
                tokens_done += int(lengths[b].sum())
                now = time.time()
                if now - last_log > 15:
                    rate = (tokens_done) / max(now - t0, 1e-9)
                    print(f"  {label}: {int(done.sum())}/{n} texts, {tokens_done}/{total_tokens} tokens, {rate:,.0f} tok/s", file=sys.stderr)
                    last_log = now
                if resume_path is not None and now - last_save > 120:
                    self._save_resume(resume_path, out, done, text_hash)
                    last_save = now
        if resume_path is not None and resume_path.exists():
            resume_path.unlink()
        return out

    def _retry_unpadded(self, ids: list[list[int]], idx: np.ndarray):
        """A padded batch produced NaN/inf (fully masked pad rows can do that on some backends). On a GPU device redo the
        batch one text at a time (no padding); on CPU, or when a single text is still bad, stop."""
        torch = self.torch
        if not self.single_fallback:
            raise RuntimeError("non-finite embeddings (NaN/inf): check dtype/device and the weights")
        self.n_fallback_batches += 1
        print(f"  warning: non-finite values in a padded batch of {len(idx)}; redoing it one text at a time", file=sys.stderr)
        rows = []
        for i in idx:
            input_ids, mask, pos = self._tensors(ids, np.array([i]))
            r = self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1].float()
            if not bool(torch.isfinite(r).all()):
                raise RuntimeError("non-finite embeddings (NaN/inf) even without padding: check dtype/device and the weights")
            rows.append(r)
        return torch.cat(rows)

    def _save_resume(self, path: Path, out: np.ndarray, done: np.ndarray, text_hash: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, emb=out, done=done, text_hash=text_hash, device=self.device)
        os.replace(tmp, path)

    # --- the read pass ---

    def acts(self, texts: list[str], max_len: int, layers: list[int]) -> dict[int, np.ndarray]:
        """Last-token residual stream after k blocks, per requested k, shape (len(texts), hidden), float32."""
        torch = self.torch
        n_layers = len(self.model.layers)
        bad = [k for k in layers if not 0 <= k <= n_layers]
        if bad:
            raise ValueError(f"layers out of range 0..{n_layers}: {bad}")
        ids = self.tokenize(texts, max_len)
        lengths = np.array([len(x) for x in ids])
        result = {k: np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32) for k in layers}
        captured: dict[int, object] = {}
        handles = []

        def make_hook(k: int):
            def hook(_module, _inputs, output):
                t = output[0] if isinstance(output, tuple) else output
                captured[k] = t[:, -1].detach().float().cpu().numpy()

            return hook

        for k in layers:
            module = self.model.embed_tokens if k == 0 else self.model.layers[k - 1]
            handles.append(module.register_forward_hook(make_hook(k)))
        try:
            with torch.inference_mode():
                for b in self._batches(lengths):
                    input_ids, mask, pos = self._tensors(ids, b)
                    self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos)
                    for k in layers:
                        result[k][b] = captured[k]
        finally:
            for h in handles:
                h.remove()
        return result
