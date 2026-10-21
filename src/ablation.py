"""GBM ablation: how much of the gradient-boosting score comes from the elapsed-cycle feature?"""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from rul import CAP, CONDITIONS, ROOT, WINDOW, load, normalise, rmse, stats, test_windows, train_windows

out = {}
for subset in ["FD001", "FD002", "FD003", "FD004"]:
    train, test, rul = load(ROOT / "data", subset)
    units = train["unit"].unique()
    val_units = set(np.random.default_rng(0).choice(units, size=round(0.2 * len(units)), replace=False))
    fit_mask = ~train["unit"].isin(val_units)
    normalise(train, test, CONDITIONS[subset], fit_mask)
    x, y, u, cyc = train_windows(train)
    tr = ~np.isin(u, list(val_units))
    xte, cte = test_windows(test)
    yte = np.minimum(rul, CAP)
    row = {}
    for name, keep in [("with_cycle", None), ("without_cycle", slice(0, -1))]:
        ftr, fte = stats(x[tr], cyc[tr]), stats(xte, cte)
        if keep is not None:
            ftr, fte = ftr[:, keep], fte[:, keep]
        gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False, random_state=0)
        row[name] = rmse(np.clip(gbm.fit(ftr, y[tr]).predict(fte), 0, CAP), yte)
    out[subset] = row
    print(subset, {k: round(v, 2) for k, v in row.items()}, flush=True)
(ROOT / "results" / "ablation_gbm_cycle.json").write_text(json.dumps(out, indent=2))
