import os, csv, json, time
from multiprocessing import Pool
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, average_precision_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of src/)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
PROC = os.path.join(DATA, "processed")
IMG_ROOT = os.path.join(RAW, "Phish360")
SPLITS = os.path.join(DATA, "splits.json")
MARKUP_DIR = os.path.join(PROC, "markup")
OUT = os.path.join(ROOT, "outputs")
RESULTS = os.path.join(OUT, "results.csv")
SUMMARY = os.path.join(OUT, "summary.csv")
PREDS = os.path.join(OUT, "preds")
CHECKPOINTS = os.path.join(OUT, "checkpoints")
BERT = "distilbert-base-uncased"
MARKUPLM = "microsoft/markuplm-base"
MARKUP_KEYS = ["input_ids", "attention_mask", "xpath_tags_seq", "xpath_subs_seq"]
MAX_LEN = 256
URL_LEN = 200
IMG_SIZE = 224
VAL_FRAC = 0.1
SPLIT_SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
URL_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
CHAR2IDX = {c: i + 2 for i, c in enumerate(URL_CHARS)}  # 0 = PAD, 1 = UNK
SECOND_LEVEL = {"co", "com", "net", "org", "gov", "edu", "ac", "or", "ne", "go"}
RESULT_FIELDS = ["time", "model", "backbone", "crop", "seed", "epochs", "best_epoch", "val_f1",
                 "acc", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp", "note"]


def domain(url):
    """Registered domain: www.login.paypal.co.uk -> paypal.co.uk"""
    host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    if host.replace(".", "").isdigit():
        return host
    parts = host.split(".")
    if len(parts) > 2 and parts[-2] in SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def load_df(html=False):
    cols = ["URL", "BeautifulSoup_text", "image_path", "split"] + (["full_html"] if html else [])
    parts = [pd.read_parquet(os.path.join(PROC, fn), columns=cols).assign(label=lab)
             for fn, lab in (("Phish360_phish.parquet", 1), ("Phish360_legit.parquet", 0))]
    df = pd.concat(parts, ignore_index=True)
    df["id"] = df.image_path.str.split("/").str[:2].str.join("/")
    df["URL"] = df.URL.fillna("")
    df["text"] = df.BeautifulSoup_text.fillna("").str[:5000]
    df["domain"] = df.URL.map(domain)
    return df.sort_values("id").reset_index(drop=True)  # fixed order: caches are indexed by row


def get_splits(df):
    """Official test set; validation = 10 % of trainval grouped by domain. Saved once, reused by all runs."""
    if not os.path.exists(SPLITS):
        tv = df[df.split == "trainval"]
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SPLIT_SEED)
        tr, va = next(gss.split(tv, tv.label, groups=tv.domain))
        sp = {"train": tv.id.iloc[tr].tolist(), "val": tv.id.iloc[va].tolist(),
              "test": df.id[df.split == "test"].tolist()}
        with open(SPLITS, "w") as f:
            json.dump(sp, f)
    with open(SPLITS) as f:
        sp = json.load(f)
    pos = pd.Series(np.arange(len(df)), index=df.id)
    return {k: pos[v].values for k, v in sp.items()}


def encode_url(url):
    ids = [CHAR2IDX.get(c, 1) for c in url.lower()[:URL_LEN]]
    return torch.tensor(ids + [0] * (URL_LEN - len(ids)))


def _load_img(args):
    path, crop = args
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.crop((0, 0, w, min(h, int(w * crop))))
    return np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)


def image_cache(df, crop):
    """Top crop (height = crop * width) resized to 224x224, cached once as a uint8 .npy memmap."""
    fn = os.path.join(PROC, f"img{IMG_SIZE}_crop{crop}.npy")
    if not os.path.exists(fn):
        print(f"building image cache {fn} (one-time) ...")
        tmp = fn + ".tmp"
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(df), IMG_SIZE, IMG_SIZE, 3))
        jobs = [(os.path.join(IMG_ROOT, p), crop) for p in df.image_path]
        with Pool() as pool:
            for i, a in enumerate(pool.imap(_load_img, jobs, chunksize=32)):
                arr[i] = a
        arr.flush()
        del arr
        os.replace(tmp, fn)
    return fn


class PhishDS(Dataset):
    def __init__(self, df, idx, mods, tok=None, img_file=None, train=False):
        self.idx, self.mods, self.img_file = np.asarray(idx), mods, img_file
        self.y = torch.tensor(df.label.values[self.idx])
        if "text" in mods:
            enc = tok(df.text.values[self.idx].tolist(), max_length=MAX_LEN, truncation=True,
                      padding="max_length", return_tensors="pt")
            self.ids, self.mask = enc.input_ids, enc.attention_mask
        if "url" in mods:
            self.url = torch.stack([encode_url(u) for u in df.URL.values[self.idx]])
        if "markup" in mods and not os.path.exists(os.path.join(MARKUP_DIR, "input_ids.npy")):
            raise SystemExit("markup cache missing - run: python src/prep_markup.py")
        self.img = self.mk = None  # memmaps are opened lazily inside each worker
        jit = [transforms.ColorJitter(0.2, 0.2, 0.2)] if train else []
        self.tf = transforms.Compose([*jit, transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j, out = self.idx[i], {"label": self.y[i]}
        if "text" in self.mods:
            out["ids"], out["mask"] = self.ids[i], self.mask[i]
        if "url" in self.mods:
            out["url"] = self.url[i]
        if "image" in self.mods:
            if self.img is None:
                self.img = np.load(self.img_file, mmap_mode="r")
            out["img"] = self.tf(torch.from_numpy(np.array(self.img[j])).permute(2, 0, 1).float() / 255)
        if "markup" in self.mods:
            if self.mk is None:
                self.mk = {k: np.load(os.path.join(MARKUP_DIR, k + ".npy"), mmap_mode="r") for k in MARKUP_KEYS}
            for k in MARKUP_KEYS:
                out[k] = torch.from_numpy(self.mk[k][j].astype(np.int64))
        return out


def metrics(y, prob, thr=0.5):
    y, prob = np.asarray(y), np.asarray(prob)
    p = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel().tolist()
    both = len(set(y.tolist())) > 1
    return {"acc": accuracy_score(y, p), "f1": f1_score(y, p, zero_division=0),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
            "roc_auc": roc_auc_score(y, prob) if both else float("nan"),
            "pr_auc": average_precision_score(y, prob) if both else float("nan"),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def fmt(m):
    return " ".join(f"{k} {m[k]:.4f}" for k in ("acc", "f1", "fpr", "fnr", "roc_auc", "pr_auc"))


def log_result(row, run_name, ids, y, prob):
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), **row}
    new = not os.path.exists(RESULTS)
    os.makedirs(OUT, exist_ok=True)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
    os.makedirs(PREDS, exist_ok=True)
    pd.DataFrame({"id": ids, "label": y, "prob_phish": prob}).to_csv(
        os.path.join(PREDS, run_name + ".csv"), index=False)
