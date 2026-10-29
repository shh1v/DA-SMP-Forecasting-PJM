"""
Concise feature-importance summary for the presentation.

Two horizontal bar charts side by side, both sorted and sharing the same
feature order (by SHAP), so the eye can compare the two methods at a glance:

  LEFT  : mean |SHAP|          -> average $/MWh each feature moves a prediction
  RIGHT : permutation importance -> MAE increase ($/MWh) when the feature is shuffled

Reads:  outputs/explain/shap_importance.csv
        outputs/explain/permutation_importance.csv
Saves:  outputs/importance_summary.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPLAIN = HERE / "explain"

# readable labels for the slide
PRETTY = {
    "smp_lag1": "Price yesterday (smp_lag1)",
    "tight": "System tightness (tight)",
    "dnl": "Net-load change (dnl)",
    "g_comp": "Gas price (g_comp)",
    "nl": "Net load (nl)",
    "ramp": "Hourly ramp (ramp)",
    "gas_basis_ew": "Gas E-W spread (gas_basis_ew)",
    "smp_lag7": "Price last week (smp_lag7)",
    "hour": "Hour of day (hour)",
    "month": "Month (month)",
    "coal_share_7": "Coal share (coal_share_7)",
    "dow": "Day of week (dow)",
    "s_east": "East load share (s_east)",
    "nl_max": "Daily peak net load (nl_max)",
    "spr_ew_lag1": "Congestion spread (spr_ew_lag1)",
    "ci_lag1": "Congestion intensity (ci_lag1)",
    "mlf_ew_lag1": "Loss spread (mlf_ew_lag1)",
    "ren_share": "Renewable share (ren_share)",
    "is_holiday": "Holiday flag (is_holiday)",
}

# ----------------------------------------------------------------- load data
shap = pd.read_csv(EXPLAIN / "shap_importance.csv", index_col=0)["mean_abs_shap"]
perm = pd.read_csv(EXPLAIN / "permutation_importance.csv").set_index("feature")["mae_increase"]

# order by SHAP (descending); keep the same order on both panels
order = shap.sort_values(ascending=True).index.tolist()        # ascending -> largest on top with barh
shap = shap.reindex(order)
perm = perm.reindex(order)
labels = [PRETTY.get(f, f) for f in order]

C_SHAP = "#4C78A8"
C_PERM = "#E45756"
C_NEG = "#BAB0AC"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
fig.suptitle("Feature importance: two independent methods agree",
             fontsize=16, fontweight="bold", y=0.98)

# ---- LEFT: SHAP ----
axL.barh(labels, shap.to_numpy(), color=C_SHAP)
axL.set_title("SHAP\n(avg $/MWh each feature moves a prediction)", fontsize=11)
axL.set_xlabel("mean |SHAP|  ($/MWh)")
for y, v in enumerate(shap.to_numpy()):
    axL.text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=8, color="#333333")
axL.set_xlim(0, shap.max() * 1.18)
axL.grid(axis="x", color="#EEEEEE")
axL.set_axisbelow(True)

# ---- RIGHT: permutation (highlight the negative bar) ----
colors = [C_NEG if v < 0 else C_PERM for v in perm.to_numpy()]
axR.barh(labels, perm.to_numpy(), color=colors)
axR.set_title("Permutation importance\n(MAE increase when the feature is shuffled)", fontsize=11)
axR.set_xlabel("MAE increase  ($/MWh)")
for y, v in enumerate(perm.to_numpy()):
    off = 0.05 if v >= 0 else -0.05
    ha = "left" if v >= 0 else "right"
    axR.text(v + off, y, f"{v:+.2f}", va="center", ha=ha, fontsize=8, color="#333333")
axR.axvline(0, color="#888888", lw=0.8)
axR.set_xlim(perm.min() * 1.6 - 0.2, perm.max() * 1.15)
axR.grid(axis="x", color="#EEEEEE")
axR.set_axisbelow(True)

# annotate the one negative (actively unhelpful) feature
if (perm < 0).any():
    neg_feat = perm.idxmin()
    y_neg = order.index(neg_feat)
    axR.annotate("shuffling it IMPROVES\naccuracy -> not useful",
                 xy=(perm[neg_feat], y_neg), xytext=(perm.max() * 0.35, y_neg + 1.8),
                 fontsize=8.5, color="#666666", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=1))

fig.text(0.5, 0.015,
         "Both methods: yesterday's price, tightness, net-load change and gas dominate;  "
         "congestion, renewable share and the holiday flag add little.",
         ha="center", fontsize=10, style="italic", color="#444444")

fig.tight_layout(rect=[0, 0.04, 1, 0.95])
out = HERE / "importance_summary.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
