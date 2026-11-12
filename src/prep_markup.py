import os, warnings
from multiprocessing import Pool
import numpy as np
from bs4 import BeautifulSoup, Comment
from transformers import MarkupLMProcessor
import common as C

MAX_LEN = 512
DTYPES = {"input_ids": np.int32, "attention_mask": np.uint8, "xpath_tags_seq": np.uint8, "xpath_subs_seq": np.int16}
EMPTY = "<html><body><p>empty</p></body></html>"
proc = None


def init():
    global proc
    warnings.filterwarnings("ignore")
    proc = MarkupLMProcessor.from_pretrained(C.MARKUPLM)


def clean(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "svg", "noscript", "meta", "link", "head"]):
        t.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    return str(soup)


def encode(html):
    kw = dict(truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="np")
    try:
        enc, ok = proc(clean(html), **kw), True
    except Exception:
        enc, ok = proc(EMPTY, **kw), False
    return {k: enc[k][0].astype(DTYPES[k]) for k in C.MARKUP_KEYS}, ok


if __name__ == "__main__":
    init()  # downloads the processor once before the workers start
    df = C.load_df(html=True)
    os.makedirs(C.MARKUP_DIR, exist_ok=True)
    arrs = {k: np.lib.format.open_memmap(os.path.join(C.MARKUP_DIR, k + ".tmp.npy"), mode="w+",
                                         dtype=DTYPES[k], shape=(len(df), MAX_LEN) + (() if "xpath" not in k else (50,)))
            for k in C.MARKUP_KEYS}
    failed = 0
    with Pool(initializer=init) as pool:
        for i, (enc, ok) in enumerate(pool.imap(encode, df.full_html.fillna(""), chunksize=16)):
            for k in C.MARKUP_KEYS:
                arrs[k][i] = enc[k]
            failed += not ok
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(df)}", flush=True)
    for k, a in arrs.items():
        a.flush()
    del arrs, a
    for k in C.MARKUP_KEYS:
        os.replace(os.path.join(C.MARKUP_DIR, k + ".tmp.npy"), os.path.join(C.MARKUP_DIR, k + ".npy"))
    print(f"done | {len(df)} pages | failed to parse (replaced by empty page): {failed}")
