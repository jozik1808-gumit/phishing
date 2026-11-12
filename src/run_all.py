"""Runs every Phish360 experiment x 3 seeds. Resumable: runs already in results.csv are skipped."""
import os, subprocess, sys
import pandas as pd
import common as C

SEEDS = [42, 43, 44]
EXPERIMENTS = [  # (model, crop)
    ("text", 0.75), ("image", 0.75), ("url", 0.75),       # 1: unimodal ablation
    ("late", 0.75),                                        # 2: baseline late fusion
    ("image", 0.5), ("late", 0.5),                         # 3: top 640 px vs whole screenshot
    ("late3", 0.75), ("gated", 0.75),                      # 5: + URL, gated fusion
    ("markup", 0.75),                                      # 6: MarkupLM instead of DistilBERT
    ("xattn", 0.75),                                       # 7: cross-attention
]


def done():
    if not os.path.exists(C.RESULTS):
        return set()
    r = pd.read_csv(C.RESULTS, dtype=str)
    return set(zip(r.model, r.crop.astype(float), r.seed))


SRC = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    if subprocess.call([sys.executable, os.path.join(SRC, cmd[0]), *cmd[1:]]) != 0:
        print("!! failed:", " ".join(cmd), flush=True)


if __name__ == "__main__":
    if not any(m.startswith("clip_") for m, _, _ in done()):
        run(["clip_exp.py"])                                # 4: frozen CLIP + MLP, zero-shot
    if not os.path.exists(os.path.join(C.MARKUP_DIR, "input_ids.npy")):
        run(["prep_markup.py"])
    for seed in SEEDS:                                      # seed-major: one full set of results comes first
        for model, crop in EXPERIMENTS:
            if (model, crop, str(seed)) not in done():
                run(["train.py", "--model", model, "--crop", str(crop), "--seed", str(seed)])
    run(["summarize.py"])
