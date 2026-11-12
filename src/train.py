import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import common as C
from models import Net, SPECS

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"


def to_dev(b):
    return {k: v.to(DEV, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps, ws = [], [], []
    for b in loader:
        b = to_dev(b)
        with torch.autocast(DEV, enabled=AMP):
            out, w = model(b)
        ps.append(out.float().softmax(1)[:, 1].cpu())
        ys.append(b["label"].cpu())
        if w is not None:
            ws.append(w.float().cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), (torch.cat(ws).numpy() if ws else None)


def set_frozen(model, names, frozen):
    for n in names:
        if n in model.enc:
            for p in model.enc[n].parameters():
                p.requires_grad = not frozen
            if frozen:
                model.enc[n].eval()  # also keeps BatchNorm statistics fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="late", choices=list(SPECS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--crop", type=float, default=0.75, help="kept screenshot height / width (0.75 = whole 1280x960)")
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="N samples per split, smoke test (not logged)")
    ap.add_argument("--save", action="store_true", help="save best weights to checkpoints/")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mods = SPECS[args.model][0]
    df = C.load_df()
    sp = C.get_splits(df)
    if args.limit:
        sp = {k: np.random.RandomState(0).permutation(v)[:args.limit] for k, v in sp.items()}
    shared = len(set(df.domain.values[sp["test"]]) & set(df.domain.values[sp["train"]]))
    print(f"{args.model} seed {args.seed} | train {len(sp['train'])} val {len(sp['val'])} "
          f"test {len(sp['test'])} | domains shared train/test: {shared}")

    tok = AutoTokenizer.from_pretrained(C.BERT) if "text" in mods else None
    img_file = C.image_cache(df, args.crop) if "image" in mods else None
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=AMP, persistent_workers=args.workers > 0)
    dl = {k: DataLoader(C.PhishDS(df, v, mods, tok, img_file, train=k == "train"),
                        shuffle=k == "train", drop_last=k == "train", **kw) for k, v in sp.items()}

    model = Net(args.model, args.backbone).to(DEV)
    counts = np.bincount(df.label.values[sp["train"]], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)),
                                                      dtype=torch.float, device=DEV))
    groups = model.param_groups()
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=args.epochs * len(dl["train"]), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=AMP)
    frozen_first = ["image"] + (["text"] if args.model == "xattn" else [])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device {DEV} | params {n_params:.1f}M | class counts {counts.tolist()} | steps/epoch {len(dl['train'])}")

    best, best_ep, best_state = -1.0, 0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        set_frozen(model, frozen_first, ep == 1)
        total = 0.0
        for step, b in enumerate(dl["train"], 1):
            b = to_dev(b)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV, enabled=AMP):
                loss = loss_fn(model(b)[0], b["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 100 == 0:
                print(f"  ep {ep} step {step}/{len(dl['train'])} loss {total / step:.4f}", flush=True)
        y, p, _ = predict(model, dl["val"])
        m = C.metrics(y, p)
        print(f"epoch {ep} | train loss {total / len(dl['train']):.4f} | val {C.fmt(m)}", flush=True)
        if m["f1"] > best:
            best, best_ep = m["f1"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y, p, w = predict(model, dl["test"])
    m = C.metrics(y, p)
    note = ""
    if w is not None:  # mean gate weight per modality, for legit and phishing pages
        note = " | ".join(f"{cls}: " + " ".join(f"{n}={w[y == c, i].mean():.3f}" for i, n in enumerate(mods))
                          for c, cls in ((0, "legit"), (1, "phish")))
        print("gate weights", note)
    print(f"TEST (best epoch {best_ep}) | {C.fmt(m)} | TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}")

    run = f"{args.model}_{args.backbone}_crop{args.crop}_s{args.seed}"
    if args.save:
        os.makedirs(C.CHECKPOINTS, exist_ok=True)
        torch.save(best_state, os.path.join(C.CHECKPOINTS, run + ".pt"))
    if not args.limit:
        C.log_result({"model": args.model, "backbone": args.backbone, "crop": args.crop, "seed": args.seed,
                      "epochs": args.epochs, "best_epoch": best_ep, "val_f1": best, "note": note, **m},
                     run, df.id.values[sp["test"]], y, p)


if __name__ == "__main__":
    main()
