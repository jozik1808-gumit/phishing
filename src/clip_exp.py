import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer
import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
PROMPTS = ["a screenshot of a legitimate website", "a screenshot of a fake phishing login page"]
FEATS = {"t": 512, "v": 512, "tv": 1024, "tvx": 2048}


def _vec(o):  # transformers v5 returns an output object, v4 a tensor
    return o if torch.is_tensor(o) else o.pooler_output


@torch.no_grad()
def compute(df, name, crop, fn):
    """Frozen CLIP: one text (77 tokens) and one image embedding per page, computed once."""
    model = CLIPModel.from_pretrained(name).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(name)
    imgs = np.load(C.image_cache(df, crop), mmap_mode="r")
    mean = torch.tensor(CLIP_MEAN, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=DEV).view(1, 3, 1, 1)
    T, V = [], []
    for s in range(0, len(df), 64):
        enc = tok(df.text.values[s:s + 64].tolist(), max_length=77, truncation=True,
                  padding="max_length", return_tensors="pt").to(DEV)
        x = torch.from_numpy(np.array(imgs[s:s + 64])).to(DEV).permute(0, 3, 1, 2).float() / 255
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            T.append(_vec(model.get_text_features(input_ids=enc.input_ids, attention_mask=enc.attention_mask)).float().cpu())
            V.append(_vec(model.get_image_features(pixel_values=(x - mean) / std)).float().cpu())
    p = tok(PROMPTS, padding=True, return_tensors="pt").to(DEV)
    P = _vec(model.get_text_features(input_ids=p.input_ids, attention_mask=p.attention_mask)).float().cpu()
    feats = {"t": torch.cat(T), "v": torch.cat(V), "prompts": P, "scale": model.logit_scale.exp().item(),
             "ids": df.id.tolist()}
    torch.save(feats, fn)
    return feats


def build(f, kind):
    t, v = F.normalize(f["t"], dim=-1), F.normalize(f["v"], dim=-1)
    return {"t": t, "v": v, "tv": torch.cat([t, v], 1), "tvx": torch.cat([t, v, t * v, (t - v).abs()], 1)}[kind]


def train_mlp(X, y, sp, seed, epochs=60):
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 2)).to(DEV)
    counts = np.bincount(y[sp["train"]], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float, device=DEV))
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)
    Xd, yd = X.to(DEV), torch.tensor(y, device=DEV)
    tr = torch.tensor(sp["train"], device=DEV)
    prob = lambda idx: mlp(Xd[idx]).softmax(1)[:, 1].detach().cpu().numpy()
    best, best_ep, state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        mlp.train()
        for b in tr[torch.randperm(len(tr), device=DEV)].split(128):
            opt.zero_grad()
            loss_fn(mlp(Xd[b]), yd[b]).backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            f1 = C.metrics(y[sp["val"]], prob(sp["val"]))["f1"]
        if f1 > best:
            best, best_ep, state = f1, ep, {k: v.clone() for k, v in mlp.state_dict().items()}
    mlp.load_state_dict(state)
    mlp.eval()
    with torch.no_grad():
        return prob(sp["test"]), best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
    ap.add_argument("--crop", type=float, default=0.75)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--feats", nargs="+", default=list(FEATS), choices=list(FEATS))
    args = ap.parse_args()

    df = C.load_df()
    sp = C.get_splits(df)
    short = args.clip.split("/")[-1]
    fn = os.path.join(C.PROC, f"clip_{short}_crop{args.crop}.pt")
    f = torch.load(fn) if os.path.exists(fn) else compute(df, args.clip, args.crop, fn)
    assert f["ids"] == df.id.tolist(), "CLIP cache order does not match data - delete " + fn
    y, ids = df.label.values, df.id.values
    print(f"CLIP {short} | embeddings {tuple(f['v'].shape)}")

    # zero-shot: cosine similarity of the screenshot with the two prompts, no training
    sims = f["scale"] * F.normalize(f["v"], dim=-1) @ F.normalize(f["prompts"], dim=-1).T
    p = sims.softmax(1)[:, 1].numpy()[sp["test"]]
    m = C.metrics(y[sp["test"]], p)
    print(f"clip_zeroshot | {C.fmt(m)}")
    C.log_result({"model": "clip_zeroshot", "backbone": short, "crop": args.crop, "seed": "", "note": " / ".join(PROMPTS), **m},
                 f"clip_zeroshot_{short}_crop{args.crop}", ids[sp["test"]], y[sp["test"]], p)

    for kind in args.feats:
        X = build(f, kind)
        for seed in args.seeds:
            p, val_f1, ep = train_mlp(X, y, sp, seed)
            m = C.metrics(y[sp["test"]], p)
            print(f"clip_{kind} seed {seed} | best epoch {ep} val f1 {val_f1:.4f} | TEST {C.fmt(m)}")
            C.log_result({"model": f"clip_{kind}", "backbone": short, "crop": args.crop, "seed": seed, "epochs": 60,
                          "best_epoch": ep, "val_f1": val_f1, **m},
                         f"clip_{kind}_{short}_crop{args.crop}_s{seed}", ids[sp["test"]], y[sp["test"]], p)


if __name__ == "__main__":
    main()
