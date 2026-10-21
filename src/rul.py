"""Remaining useful life (RUL) of turbofan engines on NASA C-MAPSS.

Protocol (standard in the literature):
- piecewise-linear target: RUL capped at 125 cycles (an engine is "healthy" early in life);
- 14 informative sensors, z-scored per operating condition (6 conditions in FD002/FD004);
- sliding windows of 30 cycles; the test prediction uses the last window of each engine;
- 20 % of the *training* engines are held out for early stopping, the test set is never used for tuning.

Models: ridge regression (linear baseline), gradient boosting on window statistics,
and a deep ensemble of 1D-CNNs with a Gaussian head (mean + variance) for calibrated uncertainty.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CAP = 125
WINDOW = 30
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
COLUMNS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
FEATURES = [f"s{i}" for i in SENSORS]
CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
Z90 = 1.6448536


def read(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df[FEATURES] = df[FEATURES].astype(np.float64)
    return df


def load(data_dir, subset):
    train = read(data_dir / f"train_{subset}.txt")
    test = read(data_dir / f"test_{subset}.txt")
    rul = np.loadtxt(data_dir / f"RUL_{subset}.txt")
    life = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = np.minimum(life - train["cycle"], CAP)
    return train, test, rul


def normalise(train, test, n_conditions, fit_mask):
    settings = ["set1", "set2", "set3"]
    if n_conditions > 1:
        km = KMeans(n_clusters=n_conditions, n_init=10, random_state=0).fit(train.loc[fit_mask, settings])
        train["cond"] = km.predict(train[settings])
        test["cond"] = km.predict(test[settings])
    else:
        train["cond"] = 0
        test["cond"] = 0
    for c in range(n_conditions):
        ref = train.loc[fit_mask & (train["cond"] == c), FEATURES]
        mu, sd = ref.mean(), ref.std().replace(0, 1)
        for df in (train, test):
            m = df["cond"] == c
            df.loc[m, FEATURES] = (df.loc[m, FEATURES] - mu) / sd


def pad(x):
    if len(x) >= WINDOW:
        return x
    return np.vstack([np.repeat(x[:1], WINDOW - len(x), axis=0), x])


def train_windows(df):
    xs, ys, units, cycles = [], [], [], []
    for unit, g in df.groupby("unit"):
        x = pad(g[FEATURES].to_numpy(np.float32))
        y = g["rul"].to_numpy(np.float32)[-len(x) + WINDOW - 1:] if len(g) >= WINDOW else g["rul"].to_numpy(np.float32)[-1:]
        cyc = g["cycle"].to_numpy()[-len(y):]
        for i in range(len(y)):
            xs.append(x[i:i + WINDOW])
        ys.extend(y)
        units.extend([unit] * len(y))
        cycles.extend(cyc)
    return np.stack(xs), np.array(ys, np.float32), np.array(units), np.array(cycles)


def test_windows(df):
    xs, cycles = [], []
    for _, g in df.groupby("unit"):
        xs.append(pad(g[FEATURES].to_numpy(np.float32))[-WINDOW:])
        cycles.append(g["cycle"].max())
    return np.stack(xs), np.array(cycles)


def stats(x, cycles):
    t = np.arange(WINDOW, dtype=np.float32) - (WINDOW - 1) / 2
    slope = (x * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.hstack([x[:, -1], x.mean(1), x.std(1), slope, cycles[:, None]])


class CNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        layers, c = [], n_features
        for _ in range(4):
            layers += [nn.Conv1d(c, 10, 10, padding="same"), nn.Tanh()]
            c = 10
        layers += [nn.Conv1d(10, 1, 3, padding="same"), nn.Tanh(), nn.Flatten(), nn.Dropout(0.5),
                   nn.Linear(WINDOW, 100), nn.Tanh(), nn.Linear(100, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x.transpose(1, 2))
        return out[:, 0], out[:, 1].clamp(-10, 4)


def fit_member(seed, xtr, ytr, xva, yva, epochs, warmup=10, patience=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN(xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr / CAP)
    xva_t = torch.from_numpy(xva)
    best, best_state, wait = math.inf, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr_t))
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            mu, logvar = model(xtr_t[idx])
            y = ytr_t[idx]
            if epoch < warmup:
                loss = ((mu - y) ** 2).mean()
            else:
                loss = (0.5 * (logvar + (y - mu) ** 2 / logvar.exp())).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model(xva_t)
        rmse = float(np.sqrt(((mu.numpy() * CAP - yva) ** 2).mean()))
        if epoch >= warmup and rmse < best - 1e-3:
            best, best_state, wait = rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        elif epoch >= warmup:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, epoch + 1


def ensemble_predict(models, x):
    with torch.no_grad():
        outs = [m(torch.from_numpy(x)) for m in models]
    mus = np.stack([o[0].numpy() * CAP for o in outs])
    variances = np.stack([o[1].exp().numpy() * CAP ** 2 for o in outs])
    mu = mus.mean(0)
    var = (variances + mus ** 2).mean(0) - mu ** 2
    return np.clip(mu, 0, None), np.sqrt(var), np.sqrt(variances.mean(0)), mus.std(0)


def nasa_score(pred, true):
    d = pred - true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def run(subset, data_dir, members, epochs):
    t0 = time.time()
    train, test, rul = load(data_dir, subset)
    units = train["unit"].unique()
    rng = np.random.default_rng(0)
    val_units = set(rng.choice(units, size=round(0.2 * len(units)), replace=False))
    fit_mask = ~train["unit"].isin(val_units)
    normalise(train, test, CONDITIONS[subset], fit_mask)

    x, y, u, cyc = train_windows(train)
    is_val = np.isin(u, list(val_units))
    xtr, ytr, xva, yva = x[~is_val], y[~is_val], x[is_val], y[is_val]
    xte, cte = test_windows(test)
    yte = np.minimum(rul, CAP)

    out = {"subset": subset, "train_engines": int(len(units) - len(val_units)), "val_engines": len(val_units),
           "test_engines": int(len(rul)), "train_windows": int(len(xtr)), "models": {}}

    ridge = Ridge(alpha=1.0).fit(xtr.reshape(len(xtr), -1), ytr)
    p = np.clip(ridge.predict(xte.reshape(len(xte), -1)), 0, CAP)
    out["models"]["ridge"] = {"rmse": rmse(p, yte), "rmse_uncapped_truth": rmse(p, rul), "score": nasa_score(p, yte)}

    gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=True,
                                        validation_fraction=None, random_state=0)
    gbm.set_params(early_stopping=False)
    gbm.fit(stats(xtr, cyc[~is_val]), ytr)
    p = np.clip(gbm.predict(stats(xte, cte)), 0, CAP)
    out["models"]["gbm"] = {"rmse": rmse(p, yte), "rmse_uncapped_truth": rmse(p, rul), "score": nasa_score(p, yte)}

    models, member_rows = [], []
    for seed in range(members):
        m, val_rmse, n_epochs = fit_member(seed, xtr, ytr, xva, yva, epochs)
        models.append(m)
        mu_single, _, _, _ = ensemble_predict([m], xte)
        member_rows.append({"seed": seed, "val_rmse": val_rmse, "epochs": n_epochs, "test_rmse": rmse(mu_single, yte)})
    mu, sd, aleatoric, epistemic = ensemble_predict(models, xte)
    err = mu - yte
    inside = np.abs(err) <= Z90 * sd
    out["models"]["cnn_ensemble"] = {
        "rmse": rmse(mu, yte),
        "rmse_uncapped_truth": rmse(mu, rul),
        "score": nasa_score(mu, yte),
        "single_member_rmse_mean": float(np.mean([r["test_rmse"] for r in member_rows])),
        "single_member_rmse_std": float(np.std([r["test_rmse"] for r in member_rows])),
        "coverage_90": float(inside.mean()),
        "mean_sigma": float(sd.mean()),
        "mean_aleatoric_sigma": float(aleatoric.mean()),
        "mean_epistemic_sigma": float(epistemic.mean()),
        "spearman_sigma_abs_error": float(spearmanr(sd, np.abs(err)).statistic),
        "members": member_rows,
    }

    mu_va, sd_va, _, _ = ensemble_predict(models, xva)
    out["models"]["cnn_ensemble"]["val_rmse_all_windows"] = rmse(mu_va, yva)
    out["seconds"] = round(time.time() - t0, 1)

    pred = {"test": {"true_capped": yte.tolist(), "true": rul.tolist(), "mu": mu.tolist(), "sigma": sd.tolist()}}
    val_unit = sorted(val_units)[0]
    sel = u[is_val] == val_unit
    pred["val_engine"] = {"unit": int(val_unit), "cycle": cyc[is_val][sel].tolist(), "target": yva[sel].tolist(),
                          "mu": mu_va[sel].tolist(), "sigma": sd_va[sel].tolist()}
    return out, pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    torch.set_num_threads(8)
    (ROOT / "results").mkdir(exist_ok=True)
    for subset in args.subsets:
        metrics, preds = run(subset, args.data, args.members, args.epochs)
        (ROOT / "results" / f"{subset}.json").write_text(json.dumps(metrics, indent=2))
        (ROOT / "results" / f"{subset}_predictions.json").write_text(json.dumps(preds))
        m = metrics["models"]
        print(subset, {k: round(v["rmse"], 2) for k, v in m.items()},
              "cov90", round(m["cnn_ensemble"]["coverage_90"], 3), f"{metrics['seconds']}s", flush=True)
