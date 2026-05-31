# make_diabetes_iid_partitions.py
# Create stratified IID-style client partitions for the diabetes HFL experiment.
# The script uses only the training indices from each fold and saves K-client
# partitions for the requested values of K.

import argparse, os, json
import numpy as np

def stratified_k_partition(indices: np.ndarray, y: np.ndarray, K: int, seed: int):
    """Split training indices into K disjoint stratified client partitions."""
    rng = np.random.default_rng(seed)
    indices = indices.astype(np.int64)
    y_sub = y[indices]

    idx0 = indices[y_sub == 0]
    idx1 = indices[y_sub == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    parts = [[] for _ in range(K)]
    for cls_idx in (idx0, idx1):
        chunks = np.array_split(cls_idx, K)
        for k in range(K):
            parts[k].extend(chunks[k].tolist())

    for k in range(K):
        rng.shuffle(parts[k])
    return parts

def summarize(parts, y):
    sizes = [len(p) for p in parts]
    pos_rates = [float(y[np.array(p, dtype=np.int64)].mean()) if len(p) else 0.0 for p in parts]
    return {
        "sizes": sizes,
        "min_size": int(min(sizes)),
        "max_size": int(max(sizes)),
        "mean_size": float(np.mean(sizes)),
        "pos_rate_mean": float(np.mean(pos_rates)),
        "pos_rate_std": float(np.std(pos_rates)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="diabetes_vfl_cv.npz")
    ap.add_argument("--out_dir", default="partitions_iid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--Ks", nargs="+", type=int, default=[10, 20])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    y = d["y"].astype(np.int64)
    folds = list(d["folds"])

    for K in args.Ks:
        out = {"npz": os.path.abspath(args.npz), "K": int(K), "seed": int(args.seed), "use": "train", "folds": []}
        out_path = os.path.join(args.out_dir, f"partitions_K{K}_train.json")

        for fold_i, f in enumerate(folds, 1):
            tr = f["train"].astype(np.int64)

            parts = stratified_k_partition(tr, y, K=K, seed=args.seed + 1000 * fold_i + K)
            summ = summarize(parts, y)

            out["folds"].append({
                "fold": int(fold_i),
                "base_indices": tr.tolist(),
                "client_indices": parts,
                "summary": summ,
            })

            print(f"[IID K={K}] fold {fold_i}: "
                  f"size min/mean/max = {summ['min_size']}/{summ['mean_size']:.1f}/{summ['max_size']}, "
                  f"pos_rate mean±std = {summ['pos_rate_mean']:.4f}±{summ['pos_rate_std']:.4f}")

        with open(out_path, "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
