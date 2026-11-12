import pandas as pd
import common as C

COLS = ["acc", "f1", "fpr", "fnr", "roc_auc", "pr_auc"]

if __name__ == "__main__":
    r = pd.read_csv(C.RESULTS)
    g = r.groupby(["model", "backbone", "crop"], sort=False)
    out = g[COLS].agg(lambda s: f"{s.mean():.4f} ± {s.std():.4f}" if len(s) > 1 else f"{s.mean():.4f}")
    out.insert(0, "runs", g.size())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(out.sort_values("f1", ascending=False).to_string())
    out.to_csv(C.SUMMARY)
    gates = r[r.note.notna() & r.model.eq("gated")]
    for _, row in gates.iterrows():
        print(f"gated seed {row.seed}: {row.note}")
