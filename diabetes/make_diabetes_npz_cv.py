import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/data/diabetes_prediction_dataset.csv")
    ap.add_argument("--out_npz", default="./diabetes_vfl_cv.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n0 = len(df)

    # remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    n1 = len(df)
    print(f"[INFO] rows: {n0} -> {n1} after drop_duplicates (removed {n0-n1})")

    # stable ids (used for alignment)
    df["ids"] = np.arange(len(df)).astype(str)

    # label
    if "diabetes" not in df.columns:
        raise ValueError("Expected column 'diabetes' in CSV")
    y = df["diabetes"].astype(int).to_numpy()

    # one-hot categorical
    cat_cols = ["gender", "smoking_history"]
    for c in cat_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in CSV")
    df_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)

    # numeric
    num_cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
    for c in num_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in CSV")
    df_num = df[num_cols].copy()

    feats_all = pd.concat([df_num, df_cat], axis=1)

    # vertical split: client X (active) + client Y (passive)
    x1_cols = ["age", "hypertension", "heart_disease"] + [c for c in feats_all.columns if c.startswith("gender_")]
    x2_cols = ["bmi", "HbA1c_level", "blood_glucose_level"] + [c for c in feats_all.columns if c.startswith("smoking_history_")]

    X1 = feats_all[x1_cols].to_numpy(dtype=np.float32)
    X2 = feats_all[x2_cols].to_numpy(dtype=np.float32)
    ids = df["ids"].to_numpy()

    # 5-fold outer CV + stratified val split inside train
    skf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    folds = []

    for outer_i, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        y_trainval = y[trainval_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed + outer_i)
        tr_rel, va_rel = next(sss.split(np.zeros(len(trainval_idx)), y_trainval))
        train_idx = trainval_idx[tr_rel]
        val_idx = trainval_idx[va_rel]

        folds.append({
            "train": train_idx.astype(np.int64),
            "val": val_idx.astype(np.int64),
            "test": test_idx.astype(np.int64),
        })

    meta = {
        "x1_cols": x1_cols,
        "x2_cols": x2_cols,
        "all_cols": list(feats_all.columns),
        "n_samples": int(len(y)),
        "pos_rate": float(y.mean()),
        "seed": int(args.seed),
        "outer_folds": int(args.outer_folds),
        "val_frac": float(args.val_frac),
    }

    np.savez(
        args.out_npz,
        X1=X1,
        X2=X2,
        y=y.astype(np.int64),
        ids=ids.astype(str),
        folds=np.array(folds, dtype=object),
        meta=json.dumps(meta),
    )

    print(f"[OK] wrote: {args.out_npz}")
    print("[INFO] X1 shape:", X1.shape, "X2 shape:", X2.shape, "y shape:", y.shape)
    print("[INFO] pos_rate:", meta["pos_rate"])
    print("[INFO] X1 cols:", x1_cols)
    print("[INFO] X2 cols:", x2_cols)

if __name__ == "__main__":
    main()
