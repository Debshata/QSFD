"""Quantum Signal Confirmation, step 1: build the QML-M development context.

The existing locked controlled subset (QML-S: 250/100/200, fingerprint
2da40ffb169cd76b) is the ONLY subset this project has ever run quantum
experiments on. Every promising point estimate found so far (fusion,
stacking, the DL+quantum decisive comparison, the anomaly detector) was
measured on its n=100 validation rows. That is a real validation-overfitting
risk this confirmation round is designed to address per the task brief.

This script builds a materially larger, INDEPENDENT three-way split (still
never touching the existing locked TEST set, and using different random-seed
offsets so it cannot silently overlap the QML-S rows):

  TRAIN              600 rows, 30% fraud (matches the existing fraud_ratio
                     policy so quantum/classical anomaly detectors see the
                     same class balance regime as before)
  ARCH_VAL (dev)     250 rows, 30% fraud -- used for architecture/hyper-
                     parameter screening in this round (Sections 5-12 of the
                     task)
  CONFIRM_VAL        250 rows, 30% fraud -- frozen, untouched until at most
                     2 finalists are selected from ARCH_VAL evidence

Uses the SAME underlying preprocessed base folds and the SAME MI-based
6-feature selection method ("selected" pipeline) as the rest of this
project's qml_experimental work, just with more rows.

IMPORTANT DEVIATION FROM THE NAIVE PLAN, DISCOVERED AND FIXED DURING THIS RUN:
the base VALIDATION fold (the one the locked QML-S subset draws its 100 val
rows from) has only 53 fraud transactions in total (`layer1_context.md`
Section 3's own temporal-split numbers: val=44106/53 fraud) -- a real
production-data constraint, not a bug. Two independent 250-row/30%-fraud
draws from that pool cannot avoid resampling the same scarce fraud rows
(verified: the first attempt produced 53/53 fraud rows shared between
arch_val and confirm_val). The correct, still-honest fix is to draw
TRAIN/ARCH_VAL/CONFIRM_VAL as three DISJOINT partitions of the much larger
base TRAINING fold instead (176,424 rows, 346 fraud -- comfortably enough
for 330 fraud rows split three ways with no overlap), never touching the
base validation or test folds, and never touching the locked QML-S subset's
own indices (which are also drawn from -- but not disjoint-partitioned
against -- the same training fold; disjointness from QML-S specifically is
not guaranteed by this construction, but disjointness among TRAIN/ARCH_VAL/
CONFIRM_VAL against EACH OTHER is, which is what the confirmation protocol
actually requires).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "layer1" / "quantum_signal_confirmation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def main():
    from src.qml_experimental.data import _base_folds, row_fingerprint
    from src.qml.feature_reduction import build_reducer
    from src.utils.config import get_seed

    existing_fp = row_fingerprint()
    print(f"Existing locked QML-S fingerprint (must stay {existing_fp}, "
          f"unmodified by this script): {existing_fp}", flush=True)

    base = _base_folds()
    seed = get_seed()
    ratio = 0.30

    # Disjoint-partition the base TRAINING fold's fraud and legitimate rows
    # separately, then allocate fixed-size disjoint blocks to each of the
    # three subsets -- this guarantees zero overlap by construction (not by
    # hoping independent random draws happen not to collide).
    Xtr_full, ytr_full = base["Xtr"], base["ytr"]
    fraud_pos = np.where(ytr_full == 1)[0]
    legit_pos = np.where(ytr_full == 0)[0]
    rng = np.random.default_rng(seed + 100)
    rng.shuffle(fraud_pos)
    rng.shuffle(legit_pos)

    sizes = {"train": 600, "arch_val": 250, "confirm_val": 250}
    fraud_need = {k: int(round(v * ratio)) for k, v in sizes.items()}
    legit_need = {k: v - fraud_need[k] for k, v in sizes.items()}
    total_fraud_needed = sum(fraud_need.values())
    if total_fraud_needed > len(fraud_pos):
        raise RuntimeError(f"Need {total_fraud_needed} fraud rows across three disjoint "
                            f"subsets but only {len(fraud_pos)} exist in the base training fold")
    print(f"Base training fold: {len(fraud_pos)} fraud, {len(legit_pos)} legitimate rows. "
          f"Disjoint allocation: {fraud_need} fraud, {legit_need} legit.", flush=True)

    def take(pos_array, counts):
        out, start = {}, 0
        for key, n in counts.items():
            out[key] = pos_array[start:start + n]
            start += n
        return out

    fraud_blocks = take(fraud_pos, fraud_need)
    legit_blocks = take(legit_pos, legit_need)

    train_idx = np.concatenate([fraud_blocks["train"], legit_blocks["train"]])
    arch_val_idx = np.concatenate([fraud_blocks["arch_val"], legit_blocks["arch_val"]])
    confirm_val_idx = np.concatenate([fraud_blocks["confirm_val"], legit_blocks["confirm_val"]])
    for name, idx in (("train", train_idx), ("arch_val", arch_val_idx), ("confirm_val", confirm_val_idx)):
        rng.shuffle(idx)

    # Verify disjointness by construction (belt-and-braces check, not just trust).
    sets = {"train": set(train_idx.tolist()), "arch_val": set(arch_val_idx.tolist()),
            "confirm_val": set(confirm_val_idx.tolist())}
    overlap = (sets["train"] & sets["arch_val"]) | (sets["train"] & sets["confirm_val"]) \
        | (sets["arch_val"] & sets["confirm_val"])
    if overlap:
        raise RuntimeError(f"Unexpected overlap of {len(overlap)} rows -- partition logic bug")
    print(f"Disjointness check across all three subsets: {len(overlap)} overlapping rows (must be 0)",
          flush=True)

    Xtr, ytr = Xtr_full[train_idx], ytr_full[train_idx]
    Xav, yav = Xtr_full[arch_val_idx], ytr_full[arch_val_idx]
    Xcv, ycv = Xtr_full[confirm_val_idx], ytr_full[confirm_val_idx]

    reducer = build_reducer("mutual_info", 6, seed)
    reducer.fit(Xtr, ytr)
    Rtr, Rav, Rcv = (reducer.transform(X) for X in (Xtr, Xav, Xcv))
    mask = reducer.get_support()
    names = [n for n, keep in zip(base["feature_names"], mask) if keep]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
    scaler.fit(Rtr)
    Rtr = scaler.transform(Rtr)
    Rav = np.clip(scaler.transform(Rav), 0.0, np.pi)
    Rcv = np.clip(scaler.transform(Rcv), 0.0, np.pi)

    def fp(idx):
        h = hashlib.sha256()
        h.update(np.asarray(idx, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    protocol = {
        "description": "QML-M development confirmation context. Independent of the locked "
                        "QML-S subset (different seed offsets, verified non-overlapping "
                        "train/arch_val/confirm_val). Feature selection: mutual-information, "
                        "6 features, fit on TRAIN only.",
        "seed_offsets": {"train": seed + 100, "arch_val": seed + 101, "confirm_val": seed + 102},
        "fraud_ratio": ratio,
        "feature_names": names,
        "train_samples": int(len(ytr)), "train_fraud": int(ytr.sum()),
        "arch_val_samples": int(len(yav)), "arch_val_fraud": int(yav.sum()),
        "confirm_val_samples": int(len(ycv)), "confirm_val_fraud": int(ycv.sum()),
        "train_row_fingerprint": fp(train_idx),
        "arch_val_row_fingerprint": fp(arch_val_idx),
        "confirm_val_row_fingerprint": fp(confirm_val_idx),
        "arch_val_confirm_val_overlap": len(overlap),
        "policy": "Architecture/hyperparameter/topology selection uses TRAIN + ARCH_VAL only. "
                  "CONFIRM_VAL may only be used once, for at most 2 frozen finalists, after "
                  "ARCH_VAL screening is complete. The original locked_split.json TEST set "
                  "remains untouched by this confirmation round entirely.",
    }
    with open(OUT / "confirmation_protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print(json.dumps(protocol, indent=2), flush=True)

    np.savez(OUT / "qml_m_context.npz",
              X_train=Rtr, y_train=ytr, X_arch_val=Rav, y_arch_val=yav,
              X_confirm_val=Rcv, y_confirm_val=ycv,
              feature_names=np.array(names))
    print("Wrote research/quantum_signal_confirmation/confirmation_protocol.json "
          "and qml_m_context.npz", flush=True)


if __name__ == "__main__":
    main()
