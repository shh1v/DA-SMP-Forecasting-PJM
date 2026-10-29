"""
Schematic of the walk-forward validation used in the PJM DA-SMP forecaster.

Two panels:
  (A) The OUTER loop: an expanding training window that is refit every 7 days,
      each fit predicting the next 7-day block through 2025.
  (B) A ZOOM on one single fit: the training window split in time into a
      "learn" part and the last-60-days internal hold-out used for early
      stopping, then a refit on the full window, then the prediction block.

Run:  python outputs/walkforward_illustration.py
Saves: outputs/walkforward_illustration.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# ----------------------------------------------------------------- colours
C_TRAIN = "#4C78A8"   # training (learn) data
C_VALID = "#F58518"   # internal 60-day hold-out (early stopping)
C_PRED = "#54A24B"    # prediction / walk-forward scored window
C_FULL = "#9ECAE1"    # full-window refit
GRID = "#DDDDDD"
TXT = "#222222"

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Walk-forward validation with an internal early-stopping hold-out",
             fontsize=16, fontweight="bold", y=0.98)

# ============================================================ PANEL A (outer loop)
axA = fig.add_axes([0.07, 0.56, 0.88, 0.34])
axA.set_title("A.  Outer loop: expanding training window, refit every 7 days, "
              "each fit scores the next block (walk-forward)",
              fontsize=12, loc="left", pad=10)

# A simplified timeline: start of training at x=0, forecast year spans x=24..36 (months-ish units).
train_start = 0.0
year_start = 24.0      # 2025-01-01 (training to here = 2 years of 2023-2024)
step = 1.5             # one "retrain block" width on this schematic (represents ~7 days)
n_blocks = 7           # show 7 example refits, then an ellipsis

bar_h = 0.55
y0 = 0

for i in range(n_blocks):
    pred_start = year_start + i * step
    # expanding training window: from fixed start to just before this block
    axA.add_patch(Rectangle((train_start, y0 - bar_h / 2), pred_start - train_start, bar_h,
                            facecolor=C_TRAIN, edgecolor="white", lw=0.8))
    # the last-60-days hold-out sits at the tail of the training window
    hold_w = 0.9
    axA.add_patch(Rectangle((pred_start - hold_w, y0 - bar_h / 2), hold_w, bar_h,
                            facecolor=C_VALID, edgecolor="white", lw=0.8))
    # prediction block (scored out-of-sample)
    axA.add_patch(Rectangle((pred_start, y0 - bar_h / 2), step, bar_h,
                            facecolor=C_PRED, edgecolor="white", lw=0.8))
    axA.text(train_start + 0.2, y0, f"fit {i+1}", va="center", ha="left",
             fontsize=8, color="white", fontweight="bold")
    y0 -= 1.0

# ellipsis row
axA.text((year_start + n_blocks * step) / 1.0, y0 + 0.4, "...  (continues weekly through Dec 2025)",
         va="center", ha="left", fontsize=10, style="italic", color=TXT)

# vertical line marking the start of the forecast year
axA.axvline(year_start, color="#444444", ls="--", lw=1)
axA.text(year_start, 1.1, "2025-01-01\n(forecasting begins)", ha="center", va="bottom",
         fontsize=9, color="#444444")
axA.text(train_start, 1.1, "2023-01-08\n(fixed training start)", ha="left", va="bottom",
         fontsize=9, color="#444444")

axA.set_xlim(-1, year_start + n_blocks * step + 6)
axA.set_ylim(y0 + 0.2, 1.9)
axA.axis("off")

# ============================================================ PANEL B (zoom on one fit)
axB = fig.add_axes([0.07, 0.10, 0.88, 0.34])
axB.set_title("B.  Zoom on ONE fit (e.g. the model used from 2025-06-10): "
              "how early stopping + full refit work",
              fontsize=12, loc="left", pad=10)

xs = 0.0
xend = 20.0            # end of training window (= D-1)
pred_w = 2.5
hold_w = 3.0          # the last-60-days hold-out
bar_h = 0.7

# --- row 1: the split used to CHOOSE the number of trees ---
y1 = 2.0
axB.add_patch(Rectangle((xs, y1 - bar_h / 2), (xend - hold_w) - xs, bar_h,
                        facecolor=C_TRAIN, edgecolor="white"))
axB.add_patch(Rectangle((xend - hold_w, y1 - bar_h / 2), hold_w, bar_h,
                        facecolor=C_VALID, edgecolor="white"))
axB.text((xs + xend - hold_w) / 2, y1, "learn (trees fit here)", va="center", ha="center",
         fontsize=10, color="white", fontweight="bold")
axB.text(xend - hold_w / 2, y1, "last 60 days\n(internal hold-out:\nearly stopping)",
         va="center", ha="center", fontsize=8.5, color="white", fontweight="bold")
axB.text(xs, y1 + 0.55, "Step 1  -  add trees until the 60-day hold-out error stops "
         "improving for 100 rounds -> best # of trees", fontsize=9.5, color=TXT)

# --- row 2: the full-window refit at the chosen tree count ---
y2 = 0.6
axB.add_patch(Rectangle((xs, y2 - bar_h / 2), xend - xs, bar_h,
                        facecolor=C_FULL, edgecolor="white"))
axB.text((xs + xend) / 2, y2, "refit on the FULL window at the best # of trees "
         "(uses every day, incl. the 60)", va="center", ha="center",
         fontsize=9.5, color="#08306B", fontweight="bold")
axB.text(xs, y2 + 0.55, "Step 2  -  refit_full = true", fontsize=9.5, color=TXT)

# --- prediction block (same on both rows, to the right of D-1) ---
for yy in (y1, y2):
    axB.add_patch(Rectangle((xend + 0.4, yy - bar_h / 2), pred_w, bar_h,
                            facecolor=C_PRED, edgecolor="white"))
    axB.text(xend + 0.4 + pred_w / 2, yy, "predict\n(scored)", va="center", ha="center",
             fontsize=8.5, color="white", fontweight="bold")

# D-1 marker and arrow from fit to prediction
axB.axvline(xend + 0.2, color="#444444", ls="--", lw=1)
axB.text(xend + 0.2, 2.75, "D-1  (cut-off: nothing after this enters the fit)",
         ha="center", va="bottom", fontsize=9, color="#444444")
arrow = FancyArrowPatch((xend, y2 - 0.55), (xend + 1.6, y2 - 0.55),
                        arrowstyle="->", mutation_scale=14, color="#444444")
axB.add_patch(arrow)
axB.text(xs, y2 - 0.75, "Step 3  -  the full-window model forecasts the next days until the weekly retrain",
         fontsize=9.5, color=TXT)

axB.set_xlim(-0.5, xend + pred_w + 3)
axB.set_ylim(-0.4, 3.2)
axB.axis("off")

# ============================================================ legend
handles = [
    mpatches.Patch(color=C_TRAIN, label="Training data the trees learn from"),
    mpatches.Patch(color=C_VALID, label="Internal hold-out: last 60 days (early stopping)"),
    mpatches.Patch(color=C_FULL, label="Full-window refit at the chosen tree count"),
    mpatches.Patch(color=C_PRED, label="Prediction window = walk-forward scored (out-of-sample)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, 0.0))

out = Path(__file__).resolve().parent / "walkforward_illustration.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
