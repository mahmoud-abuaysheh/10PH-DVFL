# make_glioma_npz.py
# Builds an "aligned" NPZ for VFL simulation (2 silos) from TCGA_InfoWithGrade.csv.
#
# What you get per outer fold:
#   - train / val / test indices (5-fold stratified CV; val is a stratified split from train)
#   - vertically split feature blocks (X1, X2) aligned by patient index
#   - HFL partitions generated ONLY on the train indices:
#       * IID split for K ∈ {5,10,15,20}
#       * Dirichlet label-skew split for α ∈ {0.3, 0.1} and K ∈ {5,10,15,20}
#
# Notes:
# - Labels (y) are stored once in the NPZ (conceptually owned by the active party / server in simulation).

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

CLIENT1_COLS = [
    "Gender", "Age_at_diagnosis", "Race",
    "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16", "PIK3CA", "NF1", "PIK3R1",
]
CLIENT2_COLS = [
    "Gender", "Age_at_diagnosis", "Race",
    "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A", "IDH2", "FAT4", "PDGFRA",
]

# HFL settings (same spirit as your derma runs)
DEFAULT_KS = [5, 10, 15, 20]
DEFAULT_ALPHAS = [0.3, 0.1]


def one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot for categorical ints; keep Age numeric (scaling should be train-only at training time)."""
    out = []
    for c in cols:
        if c == "Age_at_diagnosis":
            out.append(df[[c]].astype(np.float32))
        else:
            d = pd.get_dummies(df[c].astype("int64"), prefix=c, dummy_na=False)
            out.append(d.astype(np.float32))
    return pd.concat(out, axis=1)


def _assert_finite(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name} contains NaN/Inf. First bad indices (up to 10): {bad[:10].tolist()}")


def split_iid(indices: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = indices.copy()
    rng.shuffle(idx)
    return [idx[i::k].astype(np.int64) for i in range(k)]


def split_dirichlet_label_skew(
    indices: np.ndarray,
    y: np.ndarray,
    k: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
    max_tries: int = 200,
) -> list[np.ndarray]:
    """
    Dirichlet label-skew split on given indices.
    Returns k index arrays. Best-effort to ensure each client has >= min_size.
    """
    rng = np.random.default_rng(seed)
    y_sub = y[indices]
    classes = np.unique(y_sub)

    class_to_indices = {c: indices[y_sub == c].copy() for c in classes}
    for c in classes:
        rng.shuffle(class_to_indices[c])

    parts = None
    for _ in range(max_tries):
        buckets = [[] for _ in range(k)]
        for c in classes:
            idx_c = class_to_indices[c]
            if len(idx_c) == 0:
                continue

            props = rng.dirichlet(alpha * np.ones(k))
            counts = (props * len(idx_c)).astype(int)

            # fix rounding to sum exactly
            diff = len(idx_c) - counts.sum()
            if diff != 0:
                for j in rng.choice(k, size=abs(diff), replace=True):
                    counts[j] += 1 if diff > 0 else -1
            counts = np.clip(counts, 0, None)

            start = 0
            for j in range(k):
                take = counts[j]
                if take > 0:
                    buckets[j].append(idx_c[start:start + take])
                    start += take

        parts = [
            (np.concatenate(b).astype(np.int64) if len(b) else np.zeros((0,), dtype=np.int64))
            for b in buckets
        ]
        sizes = np.array([len(p) for p in parts], dtype=int)
        if sizes.min() >= min_size:
            return parts

        # reshuffle and retry
        for c in classes:
            rng.shuffle(class_to_indices[c])

    # fallback
    return parts if parts is not None else [np.zeros((0,), dtype=np.int64) for _ in range(k)]


def _check_split_disjoint(name: str, a: np.ndarray, b: np.ndarray):
    inter = np.intersect1d(a, b)
    if inter.size != 0:
        raise ValueError(f"{name}: overlap size={inter.size}.")


def _check_fold_integrity(fold: dict, n: int):
    tr, va, te = fold["train"], fold["val"], fold["test"]

    # bounds
    for nm, idx in [("train", tr), ("val", va), ("test", te)]:
        if idx.size == 0:
            raise ValueError(f"Fold integrity: {nm} is empty.")
        if idx.min(initial=0) < 0 or idx.max(initial=-1) >= n:
            raise ValueError(f"Fold integrity: {nm} indices out of bounds.")

    # disjoint
    _check_split_disjoint("train/val", tr, va)
    _check_split_disjoint("train/test", tr, te)
    _check_split_disjoint("val/test", va, te)

    # cover all
    union = np.union1d(np.union1d(tr, va), te)
    if union.size != n:
        missing = sorted(list(set(range(n)) - set(union.tolist())))[:20]
        raise ValueError(f"Fold integrity: train+val+test does not cover all samples. Missing (up to 20): {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to TCGA_InfoWithGrade.csv")
    ap.add_argument("--out", default="glioma_aligned_vfl_hfl_cv.npz")
    ap.add_argument("--folds", type=int, default=5, help="Outer CV folds (default=5)")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction from outer-train (default=0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=str, default="5,10,15,20", help="Comma-separated K values for HFL partitions")
    ap.add_argument("--alphas", type=str, default="0.3,0.1", help="Comma-separated Dirichlet alphas")
    ap.add_argument("--min_client_size", type=int, default=10, help="Min samples per HFL client (Dirichlet best-effort)")
    args = ap.parse_args()

    Ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    df = pd.read_csv(args.csv)

    # --- Basic data hygiene: drop exact duplicate rows ---
    print("Original shape:", df.shape)
    dup = int(df.duplicated().sum())
    print("Duplicate rows:", dup)
    if dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print("After dropping duplicates:", df.shape)
    # ----------------------------------------------------

    if "Grade" not in df.columns:
        raise ValueError("Expected label column 'Grade' (0/1).")

    missing = (set(CLIENT1_COLS + CLIENT2_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    y = df["Grade"].astype(np.int64).to_numpy()
    ids = np.arange(len(df), dtype=np.int64)
    n = len(df)

    # Build vertically split feature blocks
    X1_df = one_hot(df[CLIENT1_COLS], CLIENT1_COLS)
    X2_df = one_hot(df[CLIENT2_COLS], CLIENT2_COLS)
    X1 = X1_df.to_numpy(np.float32)
    X2 = X2_df.to_numpy(np.float32)

    # Sanity checks: alignment + finiteness
    if X1.shape[0] != n or X2.shape[0] != n or y.shape[0] != n:
        raise ValueError(f"Alignment mismatch: n={n}, X1={X1.shape}, X2={X2.shape}, y={y.shape}")
    _assert_finite("X1", X1)
    _assert_finite("X2", X2)
    _assert_finite("y", y.astype(np.float32))

    # Outer CV (test per fold)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = []

    for fold_id, (outer_tr, outer_te) in enumerate(skf.split(ids, y), start=1):
        outer_tr = outer_tr.astype(np.int64)
        outer_te = outer_te.astype(np.int64)

        # Inner split: train/val from outer_tr
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.val_frac,
            random_state=args.seed + 10_000 * fold_id,
        )
        # sss.split expects X, y; we can pass indices as X placeholder
        inner_tr_rel, inner_va_rel = next(sss.split(outer_tr, y[outer_tr]))
        tr = outer_tr[inner_tr_rel].astype(np.int64)
        va = outer_tr[inner_va_rel].astype(np.int64)
        te = outer_te.astype(np.int64)

        # HFL partitions on TRAIN ONLY
        hfl = {"iid": {}, "dirichlet": {}}
        for K in Ks:
            # IID
            parts_iid = split_iid(tr, K, seed=args.seed + 1000 * fold_id + K)
            hfl["iid"][f"K{K}"] = {
                "silo1": parts_iid,
                "silo2": parts_iid,  # same indices in simulation; stored per-silo for clarity
            }

            # Dirichlet
            for a in alphas:
                parts_d = split_dirichlet_label_skew(
                    tr,
                    y,
                    K,
                    alpha=a,
                    seed=args.seed + 2000 * fold_id + int(a * 1000) + K,
                    min_size=args.min_client_size,
                )
                hfl["dirichlet"][f"K{K}_a{a}"] = {
                    "silo1": parts_d,
                    "silo2": parts_d,
                }

        fold_obj = {
            "fold": fold_id,
            "train": tr,
            "val": va,
            "test": te,
            "hfl_partitions": hfl,
        }
        _check_fold_integrity(fold_obj, n)
        folds.append(fold_obj)

    np.savez_compressed(
        args.out,
        ids=ids,
        y=y,
        X1=X1,
        X2=X2,
        X1_cols=np.array(X1_df.columns, dtype=object),
        X2_cols=np.array(X2_df.columns, dtype=object),
        folds=np.array(folds, dtype=object),
        meta=np.array(
            {
                "client1_cols_raw": CLIENT1_COLS,
                "client2_cols_raw": CLIENT2_COLS,
                "outer_folds": args.folds,
                "val_frac": args.val_frac,
                "Ks": Ks,
                "alphas": alphas,
                "min_client_size": args.min_client_size,
                "seed": args.seed,
            },
            dtype=object,
        ),
    )

    print(f"[OK] wrote {args.out}")
    print(f"  N: {n}")
    print(f"  X1: {X1.shape}  X2: {X2.shape}  y: {y.shape}  folds: {len(folds)}")
    print(f"  positives: {int(y.sum())} / {len(y)}")
    print(f"  Ks: {Ks}  alphas: {alphas}  val_frac: {args.val_frac}")
    for f in folds[:1]:
        print(f"  fold1 sizes: train={len(f['train'])} val={len(f['val'])} test={len(f['test'])}")


if __name__ == "__main__":
    main()
