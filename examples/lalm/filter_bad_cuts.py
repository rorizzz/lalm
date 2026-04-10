#!/usr/bin/env python3
"""Filter out cuts with broken/unloadable audio from a Lhotse manifest.

Usage:
    python filter_bad_cuts.py input_cuts.jsonl.gz output_cuts.jsonl.gz

Iterates over every cut, attempts to load its audio, and writes only
the successfully loaded cuts to the output manifest.
"""

import argparse
import logging
import sys

from lhotse import CutSet
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Filter cuts with bad audio from a Lhotse manifest."
    )
    parser.add_argument("input", help="Input manifest path (e.g. cuts.jsonl.gz)")
    parser.add_argument("output", help="Output manifest path (e.g. cuts_clean.jsonl.gz)")
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=4,
        help="Number of parallel workers for audio loading check (default: 4)",
    )
    args = parser.parse_args()

    logging.info(f"Loading manifest: {args.input}")
    cuts = CutSet.from_file(args.input)

    total = 0
    dropped = 0
    dropped_ids = []
    good_cuts = []

    for cut in tqdm(cuts, desc="Checking cuts"):
        total += 1
        try:
            if cut.num_channels is not None and cut.num_channels > 1:
                raise ValueError(
                    f"Multi-channel audio not supported: num_channels={cut.num_channels}"
                )
            audio = cut.load_audio()
            if audio.shape[-1] == 0:
                raise ValueError(f"Empty audio: shape={audio.shape}")
            if audio.ndim != 2 or audio.shape[0] != 1:
                raise ValueError(f"Expected mono (1, N), got shape={audio.shape}")
            good_cuts.append(cut)
        except Exception as e:
            dropped += 1
            dropped_ids.append(cut.id)
            logging.warning(f"Dropping cut {cut.id}: {e}")

    logging.info(
        f"Done. Total: {total}, kept: {total - dropped}, dropped: {dropped}"
    )

    if dropped_ids:
        logging.info(f"Dropped cut IDs: {dropped_ids}")

    clean = CutSet.from_cuts(good_cuts)
    clean.to_file(args.output)
    logging.info(f"Saved clean manifest to: {args.output}")


if __name__ == "__main__":
    main()
