"""Download the pinned SimpleQA Verified CSV into the normal Hugging Face cache, once, and print its sha256.

    cd eval
    uv run python agent_ability/web/fetch.py                  # uses HF_HOME / HF_HUB_CACHE as set
    uv run python agent_ability/web/fetch.py --cache-dir DIR  # only when HF_HOME is unset

The file is the dataset google/simpleqa-verified (MIT), pinned to one revision. It is the question set of the
web search suite (see DESIGN.md, section 6). The sha256 printed here is recorded in
data/simpleqa_verified_pilot.yaml, and data.py checks the cached file against it.

Dataset: Haas et al., "SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge",
https://arxiv.org/abs/2509.07968
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ID = "google/simpleqa-verified"
FILENAME = "simpleqa_verified.csv"
REVISION = "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_csv(cache_dir: str | Path | None = None) -> Path | None:
    """The pinned CSV when it is already in the Hugging Face cache; never touches the network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    found = try_to_load_from_cache(
        REPO_ID, FILENAME, cache_dir=cache_dir, revision=REVISION, repo_type="dataset"
    )
    return Path(found) if isinstance(found, str) else None


def fetch(cache_dir: str | Path | None = None) -> Path:
    """Download the pinned CSV (a cache hit costs nothing) and return its path in the cache."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", revision=REVISION, cache_dir=cache_dir
    )
    return Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-dir", default=None, help="Hugging Face cache directory; leave unset when HF_HOME is set"
    )
    args = parser.parse_args(argv)
    path = fetch(args.cache_dir)
    print(f"repo      {REPO_ID}")
    print(f"revision  {REVISION}")
    print(f"file      {path}")
    print(f"sha256    {sha256_file(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
