"""
analytics.py
------------
Generates a comprehensive analytic report for the selected FER2013 batch.
Reads quality_report.csv + the raw dataset and produces:
  - Console summary
  - analytics_report.png  (multi-panel dashboard)
  - analytics_summary.txt (plain-text report)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPORT_CSV  = Path("output/reports/stage2_eda/quality_report.csv")
OUT_DIR     = Path("output/reports/stage2_eda")
FULL_TRAIN  = Path("data/train")

EMOTION_COLORS = {
    "angry":    "#E74C3C",
    "disgust":  "#8E44AD",
    "fear":     "#3498DB",
    "happy":    "#F39C12",
    "neutral":  "#95A5A6",
    "sad":      "#2980B9",
    "surprise": "#27AE60",
}

FULL_COUNTS = {
    "angry": 3995, "disgust": 436, "fear": 4097,
    "happy": 7215, "neutral": 4965, "sad": 4830, "surprise": 3171,
}

DARK_BG   = "#0F0E17"
PANEL_BG  = "#1A1A2E"
PLOT_BG   = "#16213E"
SPINE_COL = "#0F3460"
WHITE     = "white"
DIM       = "#B0B0C0"


def sa(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=WHITE, labelsize=8)
    for s in ax.spines.values(): s.set_color(SPINE_COL)
    if title:   ax.set_title(title,   color=WHITE, fontsize=10, fontweight="bold", pad=7)
    if xlabel:  ax.set_xlabel(xlabel, color=DIM,   fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color=DIM,   fontsize=8)


def main():
    df = pd.read_csv(REPORT_CSV)
    emotions = sorted(df["emotion_name"].unique())
    colors   = [EMOTION_COLORS[e] for e in emotions]

    # ── Derived stats ──────────────────────────────────────────────────────
    total       = len(df)
    usable      = int(df["is_usable"].sum())
    batch_per   = df.groupby("emotion_name").size()
    mean_bright = df.groupby("emotion_name")["mean_brightness"].mean()
    std_bright  = df.groupby("emotion_name")["mean_brightness"].std()
    mean_std    = df.groupby("emotion_name")["std_dev"].mean()
    full_total  = sum(FULL_COUNTS.values())
    batch_pct   = {e: batch_per.get(e, 0) / FULL_COUNTS[e] * 100 for e in emotions}

    # ── Print console summary ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 ANALYTIC REPORT — FER2013 Batch")
    print("="*60)
    print(f"  Full dataset size   : {full_total:,} images")
    print(f"  Batch size          : {total} images")
    print(f"  Sampling ratio      : {total/full_total*100:.2f}% of training set")
    print(f"  Usable images       : {usable}/{total} ({usable/total*100:.1f}%)")
    print(f"  Flagged images      : {total-usable}")
    print(f"  Images per class    : {total//len(emotions)} (stratified)")
    print()
    print(f"  {'Class':<10} {'Batch':>6} {'Full':>7} {'Sample%':>8} {'AvgBright':>10} {'AvgStd':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*8}")
    for e in emotions:
        n = batch_per.get(e, 0)
        print(f"  {e:<10} {n:>6} {FULL_COUNTS[e]:>7,} {batch_pct[e]:>7.2f}% "
              f"{mean_bright[e]:>10.1f} {mean_std[e]:>8.1f}")
    print()
    print(f"  Brightness range    : {df['mean_brightness'].min():.1f} - {df['mean_brightness'].max():.1f}")
    print(f"  Brightness mean     : {df['mean_brightness'].mean():.1f}")
    print(f"  Brightness std      : {df['mean_brightness'].std():.1f}")
    print(f"  Contrast (avg std)  : {df['std_dev'].mean():.1f}")
    print(f"  All images 48x48    : {(df['width']==48).all() and (df['height']==48).all()}")
    print(f"  All grayscale       : {df['is_grayscale'].all()}")
    print("="*60)

    # ══════════════════════════════════════════════════════════════════════
    # DASHBOARD FIGURE  (3 rows × 3 cols)
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── [0,0] Full dataset imbalance bar ───────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    full_vals = [FULL_COUNTS[e] for e in emotions]
    bars = ax0.bar(emotions, full_vals, color=colors, edgecolor=SPINE_COL, linewidth=0.8)
    for b, v in zip(bars, full_vals):
        ax0.text(b.get_x()+b.get_width()/2, b.get_height()+60, f"{v:,}",
                 ha="center", color=WHITE, fontsize=7, fontweight="bold")
    ax0.tick_params(axis="x", rotation=30)
    sa(ax0, "Full Dataset Distribution (28,709)", "Emotion", "Images")

    # ── [0,1] Batch vs full sampling % ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    pcts = [batch_pct[e] for e in emotions]
    bars1 = ax1.bar(emotions, pcts, color=colors, edgecolor=SPINE_COL, linewidth=0.8)
    for b, v in zip(bars1, pcts):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.03,
                 f"{v:.1f}%", ha="center", color=WHITE, fontsize=7, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    sa(ax1, "Batch Sampling Rate per Class", "Emotion", "% of Class Sampled")

    # ── [0,2] Usability pie ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    wedge_data   = [usable, total - usable]
    wedge_labels = [f"Usable\n({usable})", f"Flagged\n({total-usable})"]
    wedge_colors = ["#27AE60", "#E74C3C"] if total - usable > 0 else ["#27AE60"]
    if total - usable == 0:
        wedge_data   = [usable]
        wedge_labels = [f"All Usable\n({usable})"]
    wedges, texts, autotexts = ax2.pie(
        wedge_data, labels=wedge_labels, colors=wedge_colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": WHITE, "fontsize": 9},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
    )
    for at in autotexts: at.set_color(DARK_BG); at.set_fontweight("bold")
    ax2.set_facecolor(DARK_BG)
    ax2.set_title("Batch Usability", color=WHITE, fontsize=10, fontweight="bold", pad=7)

    # ── [1,0] Mean brightness per class (bar + error) ─────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mb  = [mean_bright[e] for e in emotions]
    sb  = [std_bright[e]  for e in emotions]
    x   = np.arange(len(emotions))
    ax3.bar(x, mb, color=colors, edgecolor=SPINE_COL, linewidth=0.8, yerr=sb,
            error_kw={"ecolor": WHITE, "capsize": 3, "linewidth": 1})
    ax3.set_xticks(x); ax3.set_xticklabels(emotions, rotation=30, color=WHITE, fontsize=8)
    ax3.axhline(df["mean_brightness"].mean(), color="white", linestyle="--",
                linewidth=1, label=f"Overall mean ({df['mean_brightness'].mean():.0f})")
    ax3.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=7)
    sa(ax3, "Mean Brightness per Class (+/- Std)", "Emotion", "Mean Pixel Brightness (0-255)")

    # ── [1,1] Contrast (std_dev) violin ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    vdata = [df[df["emotion_name"]==e]["std_dev"].values for e in emotions]
    parts = ax4.violinplot(vdata, positions=range(len(emotions)), showmeans=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_alpha(0.75)
    for key in ("cmeans","cbars","cmins","cmaxes"):
        parts[key].set_color(WHITE if key=="cmeans" else "#445566")
    ax4.set_xticks(range(len(emotions)))
    ax4.set_xticklabels(emotions, rotation=30, color=WHITE, fontsize=8)
    sa(ax4, "Pixel Std Dev (Contrast) per Class", "Emotion", "Std Dev")

    # ── [1,2] Brightness scatter (brightness vs std_dev, coloured by class)
    ax5 = fig.add_subplot(gs[1, 2])
    for e, c in EMOTION_COLORS.items():
        sub = df[df["emotion_name"] == e]
        ax5.scatter(sub["mean_brightness"], sub["std_dev"],
                    color=c, alpha=0.75, s=25, label=e, edgecolors="none")
    ax5.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=7,
               markerscale=1.2, loc="upper right")
    ax5.axvline(10,  color="#E74C3C", linestyle=":", linewidth=0.8)
    ax5.axvline(245, color="#F39C12", linestyle=":", linewidth=0.8)
    sa(ax5, "Brightness vs Contrast (per image)", "Mean Brightness", "Std Dev (Contrast)")

    # ── [2,0-1] Brightness histogram (overall + per class overlay) ─────────
    ax6 = fig.add_subplot(gs[2, 0:2])
    ax6.hist(df["mean_brightness"], bins=40, color="#5B9BD5", alpha=0.6,
             edgecolor="none", label="All classes", density=True)
    for e, c in EMOTION_COLORS.items():
        sub = df[df["emotion_name"] == e]["mean_brightness"]
        ax6.hist(sub, bins=20, color=c, alpha=0.35, edgecolor="none",
                 density=True, label=e)
    ax6.axvline(df["mean_brightness"].mean(), color="white", linestyle="--",
                linewidth=1.2, label=f"mean={df['mean_brightness'].mean():.1f}")
    ax6.legend(facecolor=PANEL_BG, labelcolor=WHITE, fontsize=7,
               ncol=2, loc="upper right")
    sa(ax6, "Brightness Distribution — Batch Overview (All Classes Overlay)",
       "Mean Pixel Brightness (0-255)", "Density")

    # ── [2,2] Per-class image count in batch ──────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    counts_batch = [batch_per.get(e, 0) for e in emotions]
    wedges2, _, auto2 = ax7.pie(
        counts_batch, labels=emotions, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": WHITE, "fontsize": 7},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5},
    )
    for at in auto2: at.set_color(DARK_BG); at.set_fontsize(7); at.set_fontweight("bold")
    ax7.set_facecolor(DARK_BG)
    ax7.set_title("Batch Composition", color=WHITE, fontsize=10, fontweight="bold", pad=7)

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "FER2013 — Phase 1 Analytic Dashboard  |  Batch: 210 images  |  Real Kaggle Data",
        color=WHITE, fontsize=14, fontweight="bold", y=0.98,
    )

    out_path = OUT_DIR / "analytics_dashboard.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Dashboard saved -> {out_path.resolve()}\n")

    # ── Plain-text summary file ────────────────────────────────────────────
    txt_path = OUT_DIR / "analytics_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("FER2013 Phase 1 — Analytic Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Full training set   : {full_total:,} images\n")
        f.write(f"Batch size          : {total} images\n")
        f.write(f"Sampling ratio      : {total/full_total*100:.2f}%\n")
        f.write(f"Usable images       : {usable}/{total} ({usable/total*100:.1f}%)\n")
        f.write(f"Flagged images      : {total-usable}\n\n")
        f.write(f"{'Class':<10} {'Batch':>6} {'Full':>8} {'Sample%':>9} "
                f"{'AvgBright':>11} {'AvgContrast':>12}\n")
        f.write("-"*60 + "\n")
        for e in emotions:
            n = batch_per.get(e, 0)
            f.write(f"{e:<10} {n:>6} {FULL_COUNTS[e]:>8,} {batch_pct[e]:>8.2f}% "
                    f"{mean_bright[e]:>11.1f} {mean_std[e]:>12.1f}\n")
        f.write("\n")
        f.write(f"Brightness range    : {df['mean_brightness'].min():.1f} - {df['mean_brightness'].max():.1f}\n")
        f.write(f"Brightness mean     : {df['mean_brightness'].mean():.2f}\n")
        f.write(f"Brightness std      : {df['mean_brightness'].std():.2f}\n")
        f.write(f"Avg contrast (std)  : {df['std_dev'].mean():.2f}\n")
        f.write(f"All images 48x48    : {(df['width']==48).all() and (df['height']==48).all()}\n")
        f.write(f"All grayscale       : {df['is_grayscale'].all()}\n\n")
        f.write("Class imbalance note:\n")
        f.write(f"  Most common  : happy ({FULL_COUNTS['happy']:,} images)\n")
        f.write(f"  Least common : disgust ({FULL_COUNTS['disgust']:,} images)\n")
        f.write(f"  Imbalance ratio : {FULL_COUNTS['happy']/FULL_COUNTS['disgust']:.1f}x\n")

    print(f"  Summary saved  -> {txt_path.resolve()}\n")


if __name__ == "__main__":
    main()
