#!/usr/bin/env python3
"""
Export LexiNet pickle n-gram models into a browser-friendly JSON file.

Run from the lexinet repository root, for example:

    python3 scripts/export_lexinet_web_model.py \
      --models-dir results/models \
      --orders 3 4 5 \
      --out ../djdhillxn.github.io/assets/json/lexinet/lexinet_web_model_3_5.json

For the local n=6 model, run:

    python3 scripts/export_lexinet_web_model.py \
      --models-dir results/models \
      --orders 3 4 5 6 \
      --out ../djdhillxn.github.io/assets/json/lexinet/lexinet_web_model_3_6.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping

SPECIAL_TOKENS = {"<s>": "^", "</s>": "$"}


def keyify(context: Iterable[str]) -> str:
    """Convert tuple contexts like ('<s>', 'a', '_') to compact JS keys like '^a_'."""
    return "".join(SPECIAL_TOKENS.get(token, token) for token in context)


def compact_counter(counter: Mapping[str, int]) -> dict[str, int]:
    """Keep only non-zero integer counts, sorted for deterministic output."""
    return {str(key): int(value) for key, value in sorted(counter.items()) if int(value) > 0}


def export_model(models_dir: Path, orders: list[int]) -> dict:
    payload = {
        "version": 1,
        "alphabet": "abcdefghijklmnopqrstuvwxyz",
        "tokenMap": {"^": "<s>", "$": "</s>", "_": "unknown"},
        "notes": (
            "Browser export for the LexiNet portfolio demo. "
            "Only generation-time structures are included: ngrams, ngramsRev, and unigrams."
        ),
        "orders": {},
    }

    for n in orders:
        model_path = models_dir / f"n_{n}_gram_model_kneser_ney.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        with model_path.open("rb") as file:
            data = pickle.load(file)

        payload["orders"][str(n)] = {
            "ngrams": {
                keyify(prefix): compact_counter(counts)
                for prefix, counts in data["ngrams"].items()
            },
            "ngramsRev": {
                keyify(suffix): compact_counter(counts)
                for suffix, counts in data["ngrams_rev"].items()
            },
            "unigrams": compact_counter(data["unigrams"]),
        }
        print(
            f"n={n}: {len(data['ngrams']):,} forward contexts, "
            f"{len(data['ngrams_rev']):,} reverse contexts"
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--gzip-copy",
        action="store_true",
        help="Also write a .gz copy for size inspection or advanced serving setups.",
    )
    args = parser.parse_args()

    payload = export_model(args.models_dir, args.orders)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    args.out.write_bytes(raw)
    print(f"wrote {args.out} ({len(raw) / 1024 / 1024:.2f} MiB)")

    if args.gzip_copy:
        gz_path = args.out.with_suffix(args.out.suffix + ".gz")
        gz_path.write_bytes(gzip.compress(raw, compresslevel=9))
        print(f"wrote {gz_path} ({gz_path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
