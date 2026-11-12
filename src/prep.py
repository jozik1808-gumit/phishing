import os, re, warnings
from multiprocessing import Pool
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from common import IMG_ROOT as ROOT, PROC as OUT

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def read(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8-sig", errors="replace")


def html_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))


def parse(args):
    split, name = args
    d = os.path.join(ROOT, split, name)
    html = read(os.path.join(d, "RAW-HTML", "index.html"))
    prefix, brand = name.split("_", 1)
    return {
        "URL": read(os.path.join(d, "URL", "url.txt")).strip(),
        "BeautifulSoup_text": html_text(html),
        "image_path": f"{split}/{name}/SCREEN-SHOT/screen_shoot.png",
        "full_html": html,
        "split": split,
        "brand": "" if prefix.startswith("L") else brand,
        "label": 1 if prefix.startswith("P") else 0,
    }


if __name__ == "__main__":
    jobs = [(s, n) for s in ("trainval", "test") for n in sorted(os.listdir(os.path.join(ROOT, s)))]
    with Pool() as p:
        rows = p.map(parse, jobs, chunksize=64)
    df = pd.DataFrame(rows)
    empty = (df.full_html.str.len() == 0).sum()
    if empty:
        raise SystemExit(f"{empty} samples have empty HTML - re-extract the zip (wrong password?)")
    os.makedirs(OUT, exist_ok=True)
    for lab, fn in ((1, "Phish360_phish.parquet"), (0, "Phish360_legit.parquet")):
        df[df.label == lab].drop(columns="label").to_parquet(os.path.join(OUT, fn), index=False)
    print(df.groupby(["split", "label"]).size())
    print("empty URL:", (df.URL == "").sum(), "| empty text:", (df.BeautifulSoup_text == "").sum())
