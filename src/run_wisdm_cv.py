import os
import argparse
import numpy as np
import pandas as pd

# use GroupKFold instead of StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# fixed seeds so numbers don't jump every run
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- small metric helpers ----------
def compute_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    P = tp + fn
    N = tn + fp
    return tp, tn, fp, fn, P, N

def compute_all_metrics(y_true, y_pred, y_prob):
    tp, tn, fp, fn, P, N = compute_counts(y_true, y_pred)

    # rates
    tpr = tp / P if P else 0.0     # recall / sensitivity
    tnr = tn / N if N else 0.0     # specificity
    fpr = fp / N if N else 0.0
    fnr = fn / P if P else 0.0

    # basic scores
    acc = (tp + tn) / (P + N) if (P + N) else 0.0
    bal_acc = (tpr + tnr) / 2.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    err = 1.0 - acc

    # skill scores
    tss = tpr - fpr  # True Skill Statistic
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = (2 * (tp * tn - fn * fp)) / denom if denom else 0.0  # Heidke

    # Brier and Brier Skill
    y_prob = np.asarray(y_prob, dtype=float)
    bs = float(np.mean((y_prob - y_true) ** 2))
    p_bar = float(np.mean(y_true))
    bs_ref = p_bar * (1 - p_bar)
    bss = 1.0 - (bs / bs_ref) if bs_ref > 0 else 0.0

    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    out = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "P": P, "N": N,
        "TPR": tpr, "TNR": tnr, "FPR": fpr, "FNR": fnr,
        "Accuracy": acc, "BalancedAccuracy": bal_acc,
        "Precision": prec, "Recall": rec, "F1": f1,
        "ErrorRate": err, "TSS": tss, "HSS": hss,
        "BS": bs, "BSS": bss, "AUC": auc
    }
    return out

def plot_roc(y_true, y_prob, title, out_png):
    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    plt.figure()

    # Diagonal baseline
    plt.plot([0, 1], [0, 1], "--", color="C0")

    # If only one class is present, skip ROC and just annotate
    if len(np.unique(y_true)) < 2:
        plt.title(title + " (only one class in this fold)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="lower right")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        return

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.plot(fpr, tpr, color="C1", label=f"AUC={auc:.3f}")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ---------- small feature builder for Random Forest/SVM ----------
def window_stats(X3d):
    """
    X3d shape: (num_windows, timesteps, channels).
    For classic models just simple statistics per channel is used.
    """
    feats = []
    for w in X3d:
        f = []
        for c in range(w.shape[1]):
            v = w[:, c]
            f.extend([
                float(np.mean(v)),
                float(np.std(v)),
                float(np.min(v)),
                float(np.max(v)),
                float(np.median(v)),
            ])
        feats.append(f)
    return np.asarray(feats, dtype=float)

# ---------- sequence model ----------
def make_gru_model(tsteps, channels):
    model = Sequential()
    model.add(GRU(64, input_shape=(tsteps, channels)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------- windowing ----------
def make_windows_for_one_user(df_user, cols, win_len, stride):
    X_list, y_list = [], []
    Xarr = df_user[cols].values.astype(float)
    yarr = df_user["y"].values.astype(int)

    # majority label inside the window
    def majority(lbl):
        vals, cnt = np.unique(lbl, return_counts=True)
        return int(vals[np.argmax(cnt)])

    end = len(df_user) - win_len + 1
    for start in range(0, max(0, end), stride):
        stop = start + win_len
        X_list.append(Xarr[start:stop])
        y_list.append(majority(yarr[start:stop]))
    return X_list, y_list

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/WISDM_raw.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--sample_rate", type=float, default=20.0)   # ~20Hz in WISDM
    ap.add_argument("--win_sec", type=float, default=5.0)        # 5s windows
    ap.add_argument("--stride", type=int, default=20)            # ~1s hop at 20Hz
    ap.add_argument("--epochs", type=int, default=8)             # keep small for CPU
    ap.add_argument("--batch_size", type=int, default=64)
    # mapping to binary
    ap.add_argument("--moving", nargs="+",
                    default=["Walking", "Jogging", "Upstairs", "Downstairs", "Running"])
    ap.add_argument("--stationary", nargs="+",
                    default=["Sitting", "Standing", "Typing", "Writing", "Clapping",
                             "Drinking", "EatingSoup", "EatingChips", "EatingPasta",
                             "EatingSandwich", "BrushingTeeth"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load CSV (the builder writes these exact columns)
    df = pd.read_csv(args.csv)
    keep = ["user", "activity", "timestamp", "ax", "ay", "az"]
    df = df[keep].copy()

    # clean timestamps & order within each user
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["user", "timestamp"], kind="mergesort").reset_index(drop=True)

    # binary label
    moving = set(args.moving)
    stationary = set(args.stationary)

    def to_bin(a):
        if a in moving:
            return 1
        if a in stationary:
            return 0
        return 0

    df["y"] = df["activity"].astype(str).map(to_bin).astype(int)

    # windowing per user
    cols = ["ax", "ay", "az"]
    win_len = int(round(args.win_sec * args.sample_rate))  # samples/window
    X_all, y_all, groups_all = [], [], []   # keep user per window

    for user_id, g in df.groupby("user", sort=False):
        Xw, yw = make_windows_for_one_user(g, cols, win_len, args.stride)
        X_all += Xw
        y_all += yw
        groups_all += [user_id] * len(yw)   # group label

    X_seq = np.asarray(X_all, dtype=float)  # (N, T, C)
    y = np.asarray(y_all, dtype=int)
    groups = np.asarray(groups_all)

    print("Windows:", X_seq.shape)
    u, c = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(u, c)))
    print(f"Unique users: {len(np.unique(groups))}")
    assert len(np.unique(groups)) >= args.splits, "Need at least as many users as folds."

    # classic features for Random Forest/SVM
    print("Building features ...")
    X_tab = window_stats(X_seq)

    # models
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED)
    svm = Pipeline([("sc", StandardScaler()), ("svc", SVC(kernel="rbf", probability=True, random_state=SEED))])

    # GroupKFold for subject-wise CV
    gkf = GroupKFold(n_splits=args.splits)

    # holders for per-fold rows
    rows_rf, rows_svm, rows_gru = [], [], []

    # ----- Random Forest -----
    print("Training RandomForest ...")
    fold = 1
    for tr, te in gkf.split(X_tab, y, groups=groups):
        rf.fit(X_tab[tr], y[tr])
        yhat = rf.predict(X_tab[te])
        ypr = rf.predict_proba(X_tab[te])[:, 1]
        m = compute_all_metrics(y[te], yhat, ypr)
        m.update({"Model": "RandomForest", "Fold": fold})
        rows_rf.append(m)
        plot_roc(y[te], ypr, f"RandomForest ROC - fold {fold}", os.path.join(args.out, f"roc_RF_fold{fold}.png"))
        fold += 1

    # ----- SVM -----
    print("Training SVM ...")
    fold = 1
    for tr, te in gkf.split(X_tab, y, groups=groups):
        svm.fit(X_tab[tr], y[tr])
        yhat = svm.predict(X_tab[te])
        ypr = svm.predict_proba(X_tab[te])[:, 1]
        m = compute_all_metrics(y[te], yhat, ypr)
        m.update({"Model": "SVM", "Fold": fold})
        rows_svm.append(m)
        plot_roc(y[te], ypr, f"SVM ROC - fold {fold}", os.path.join(args.out, f"roc_SVM_fold{fold}.png"))
        fold += 1

    # ----- GRU (sequence) -----
    print("Training GRU ...")
    fold = 1
    for tr, te in gkf.split(X_seq, y, groups=groups):
        model = make_gru_model(X_seq.shape[1], X_seq.shape[2])
        # basic class weights for imbalance
        pos = float(np.mean(y[tr]))
        cw = {0: 1.0, 1: max(1.0, (1.0 - pos) / max(pos, 1e-6))}
        model.fit(X_seq[tr], y[tr], epochs=args.epochs, batch_size=args.batch_size, verbose=0, class_weight=cw)
        ypr = model.predict(X_seq[te], verbose=0).ravel()
        yhat = (ypr >= 0.5).astype(int)
        m = compute_all_metrics(y[te], yhat, ypr)
        m.update({"Model": "GRU", "Fold": fold})
        rows_gru.append(m)
        plot_roc(y[te], ypr, f"GRU ROC - fold {fold}", os.path.join(args.out, f"roc_GRU_fold{fold}.png"))
        fold += 1

    print("Saving CSVs and summary ...")
    # save per-fold
    pd.DataFrame(rows_rf).to_csv(os.path.join(args.out, "RandomForest_per_fold.csv"), index=False)
    pd.DataFrame(rows_svm).to_csv(os.path.join(args.out, "SVM_per_fold.csv"), index=False)
    pd.DataFrame(rows_gru).to_csv(os.path.join(args.out, "GRU_per_fold.csv"), index=False)

    # summary means (one row per model)
    def mean_row(df_):
        cols = [c for c in df_.columns if c not in ("Model", "Fold")]
        out = df_[cols].mean(numeric_only=True).to_dict()
        out["Model"] = df_["Model"].iloc[0]
        out["Fold"] = 0
        return out

    s = []
    s.append(mean_row(pd.DataFrame(rows_rf)))
    s.append(mean_row(pd.DataFrame(rows_svm)))
    s.append(mean_row(pd.DataFrame(rows_gru)))
    pd.DataFrame(s).to_csv(os.path.join(args.out, "summary_means.csv"), index=False)

    # print short table for screenshots
    show = ["Model", "Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1", "AUC", "BS", "BSS", "TSS", "HSS"]
    print("\n=== Summary (means) ===")
    print(pd.DataFrame(s)[show].to_string(index=False))

if __name__ == "__main__":
    main()
