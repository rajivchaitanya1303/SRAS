from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from sras.utils.io import ensure_dir, load_json
from sras.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Journal-quality style ─────────────────────────────────────────────────────
# Designed for IEEE/ACM single-column and double-column figures.
# All sizes in points; final PDFs are vector so DPI only affects raster previews.
_JOURNAL_STYLE: Dict = {
    # Typography
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "legend.title_fontsize": 9,
    # Lines & ticks
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "lines.linewidth":      1.8,
    "lines.markersize":     5,
    # Grid
    "axes.grid":            True,
    "grid.color":           "#e0e0e0",
    "grid.linewidth":       0.6,
    "grid.linestyle":       "--",
    "axes.axisbelow":       True,
    # Frame
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Output
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "pdf.fonttype":         42,      # embeds fonts as TrueType in PDF
    "ps.fonttype":          42,
}

# ── Colour palettes ───────────────────────────────────────────────────────────
# Primary accent colours (colourblind-accessible)
_BLUE     = "#2166ac"
_ORANGE   = "#d6604d"
_GREEN    = "#4dac26"
_PURPLE   = "#762a83"
_GRAY     = "#878787"
_LBLUE    = "#92c5de"   # light blue: baselines
_LORANGE  = "#f4a582"   # light orange

# SRAS-variant colours: full-colour for trained, muted for baselines
_SRAS_COLOR_MAP = {
    "ppo_base":      _BLUE,
    "supervised":    _GREEN,
    "ppo_nosw":      _PURPLE,
    "ppo_nors":      _ORANGE,
    "ppo_nocl":      "#b2df8a",
    # baselines
    "bm25":          "#bababa",
    "dense":         "#969696",
    "hybrid":        "#737373",
    "learned_ranker":"#525252",
}

_SRAS_TRAINED = {"ppo_base", "supervised", "ppo_nosw", "ppo_nors", "ppo_nocl"}

# Pretty display names
_DISPLAY_NAMES = {
    "ppo_base":       "SRAS-PPO",
    "supervised":     "SRAS-SL",
    "ppo_nosw":       "SRAS-PPO\n(no warmup)",
    "ppo_nors":       "SRAS-PPO\n(no reward shaping)",
    "ppo_nocl":       "SRAS-PPO\n(no curriculum)",
    "bm25":           "BM25",
    "dense":          "Dense",
    "hybrid":         "Hybrid",
    "learned_ranker": "Learned\nRanker",
}

# Hatching patterns (for print B&W friendliness)
_HATCH_MAP = {
    "ppo_base":       "",
    "supervised":     "//",
    "ppo_nosw":       "xx",
    "ppo_nors":       "..",
    "ppo_nocl":       "\\\\",
    "bm25":           "///",
    "dense":          "---",
    "hybrid":         "xxx",
    "learned_ranker": "...",
}


def _display(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


def _color(name: str, default: str = _GRAY) -> str:
    return _SRAS_COLOR_MAP.get(name, default)


def _hatch(name: str) -> str:
    return _HATCH_MAP.get(name, "")


class PlotGenerator:
    """
    Generates publication-quality figures for the SRAS paper.

    All figures are saved as PDF (vector) with embedded fonts, suitable for
    direct inclusion in LaTeX documents.
    """

    def __init__(self, output_dir: str = "figures/sras_model_eval") -> None:
        self.output_dir = output_dir
        ensure_dir(output_dir)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_matplotlib(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update(_JOURNAL_STYLE)
        return plt

    @staticmethod
    def _save(plt, fig, path: str) -> str:
        plt.tight_layout(pad=0.5)
        plt.savefig(path, format="pdf", bbox_inches="tight", metadata={"Creator": "SRAS"})
        plt.close(fig)
        logger.info("Saved figure: %s", path)
        return path

    @staticmethod
    def _annotate_bars(ax, bars, vals, fmt=".4f", pad_frac=0.015, fontsize=7.5):
        """Place value labels just above each bar."""
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + yrange * pad_frac,
                f"{val:{fmt}}",
                ha="center", va="bottom",
                fontsize=fontsize, color="#333333",
                clip_on=False,
            )

    @staticmethod
    def _zoom_ylim(ax, vals, lo_pad: float = 0.3, hi_pad: float = 0.2):
        """Set y-limit that zooms to [min-pad, max+pad] relative to value range."""
        if not vals:
            return
        vmin, vmax = min(vals), max(vals)
        span = max(vmax - vmin, 1e-4)
        lo = max(0.0, vmin - span * lo_pad)
        hi = vmax + span * hi_pad
        ax.set_ylim(lo, hi)

    # ── Reward curves ─────────────────────────────────────────────────────────

    def plot_reward_curves(
        self,
        log_paths: Dict[str, str],
        output_filename: str = "ppo_reward_curves.pdf",
    ) -> str:
        plt = self._get_matplotlib()
        try:
            from scipy.ndimage import uniform_filter1d
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        markers    = ["o", "s", "^", "D", "v"]

        for idx, (variant_name, log_path) in enumerate(log_paths.items()):
            if not os.path.exists(log_path):
                logger.warning("Log not found: %s", log_path)
                continue
            log = load_json(log_path)
            epochs  = [e["epoch"] for e in log]
            rewards = [e["avg_reward"] for e in log]

            # Smooth with a 3-point moving average for visual clarity
            if _has_scipy and len(rewards) >= 5:
                smoothed = uniform_filter1d(rewards, size=3, mode="nearest")
            else:
                smoothed = rewards

            ls = line_styles[idx % len(line_styles)]
            mk = markers[idx % len(markers)]
            col = _color(variant_name)
            ax.plot(
                epochs, smoothed,
                linestyle=ls, marker=mk, color=col,
                linewidth=1.8, markersize=4, markevery=max(1, len(epochs)//10),
                label=_display(variant_name), zorder=3,
            )
            # Light shaded original trace
            ax.plot(epochs, rewards, linestyle=ls, color=col, alpha=0.2, linewidth=0.8)

        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Average Reward")
        ax.set_title("PPO Training Reward Curves")
        ax.legend(framealpha=0.9, edgecolor="none", loc="lower right")

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Comparison bar (main results figure) ─────────────────────────────────

    def plot_comparison_bar(
        self,
        eval_results: Dict[str, Dict],
        metric_keys: Optional[List[str]] = None,
        output_filename: str = "comparison_bar_plot.pdf",
        dataset_label: str = "Internal",
    ) -> str:
        """
        Two-panel bar chart comparing all selectors on Relaxed-F1 and BERTScore-F1.
        SRAS-trained variants use solid colours; baselines use muted hatching.
        """
        plt = self._get_matplotlib()
        from matplotlib.patches import Patch

        if metric_keys is None:
            metric_keys = ["relaxed_f1", "bertscore_f1"]

        variants = list(eval_results.keys())

        # Build value matrix
        def _get_val(v: str, m: str) -> float:
            entry = eval_results[v]
            if isinstance(entry, dict) and "metrics" in entry:
                return entry["metrics"].get(m, 0.0)
            return entry.get(m, 0.0) if isinstance(entry, dict) else 0.0

        all_values = {m: [_get_val(v, m) for v in variants] for m in metric_keys}

        metric_labels = {
            "relaxed_f1":   "Relaxed F1",
            "bertscore_f1": "BERTScore F1",
        }
        metric_ylabels = {
            "relaxed_f1":   "Relaxed F1",
            "bertscore_f1": "BERTScore F1",
        }
        # Accent colours per metric panel
        metric_sras_color = {
            "relaxed_f1":   _BLUE,
            "bertscore_f1": _ORANGE,
        }
        metric_base_color = {
            "relaxed_f1":   _LBLUE,
            "bertscore_f1": _LORANGE,
        }

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))

        for ax, metric in zip(axes, metric_keys):
            vals   = all_values[metric]
            colors = [
                metric_sras_color[metric] if v in _SRAS_TRAINED else metric_base_color[metric]
                for v in variants
            ]
            hatches = [_hatch(v) for v in variants]
            x = np.arange(len(variants))

            bars = ax.bar(
                x, vals,
                width=0.65,
                color=colors,
                hatch=hatches,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
                zorder=2,
            )

            # Value annotations
            self._zoom_ylim(ax, vals, lo_pad=0.25, hi_pad=0.28)
            self._annotate_bars(ax, bars, vals, fmt=".4f", fontsize=7.0)

            # Best-baseline reference line
            baseline_vals = [v for vname, v in zip(variants, vals) if vname not in _SRAS_TRAINED]
            if baseline_vals:
                ax.axhline(
                    max(baseline_vals),
                    color="#555555", linewidth=0.9, linestyle=":",
                    label="Best baseline", zorder=1,
                )

            # Tick labels: pretty names, rotated
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_display(v) for v in variants],
                rotation=30, ha="right", fontsize=7.5,
            )
            ax.set_ylabel(metric_ylabels.get(metric, metric))
            ax.set_title(metric_labels.get(metric, metric), fontweight="bold")

            # Legend
            legend_elems = [
                Patch(facecolor=metric_sras_color[metric], label="SRAS variants"),
                Patch(facecolor=metric_base_color[metric], label="Baselines"),
            ]
            if baseline_vals:
                from matplotlib.lines import Line2D
                legend_elems.append(
                    Line2D([0], [0], color="#555555", linewidth=0.9, linestyle=":",
                           label="Best baseline")
                )
            ax.legend(
                handles=legend_elems, loc="lower right",
                framealpha=0.9, edgecolor="none", fontsize=7.5,
            )

        fig.suptitle(
            f"Selector Comparison: {dataset_label} Dataset",
            fontsize=10, fontweight="bold", y=1.01,
        )

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Ablation bar ──────────────────────────────────────────────────────────

    def plot_ablation_bar(
        self,
        summary: List[Dict],
        variants: List[str],
        output_filename: str = "ablation_bar_plot.pdf",
    ) -> str:
        """
        Two-panel figure: Relaxed-F1 bars (left) and BERTScore-F1 bars (right),
        both with latency annotations.
        """
        plt = self._get_matplotlib()
        from matplotlib.patches import Patch

        data_map = {e["name"]: e for e in summary if e["name"] in variants}
        ordered  = [data_map[v] for v in variants if v in data_map]
        if not ordered:
            logger.warning("No matching variants found for ablation bar.")
            return ""

        labels   = [e["name"] for e in ordered]
        f1_vals  = [e.get("relaxed_f1",   0.0) for e in ordered]
        bs_vals  = [e.get("bertscore_f1", 0.0) for e in ordered]
        lat_vals = [e.get("avg_latency_ms", 0.0) for e in ordered]
        colors   = [_color(l, default=_GRAY) for l in labels]
        hatches  = [_hatch(l) for l in labels]
        x        = np.arange(len(labels))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8))

        for ax, vals, title, ylabel in [
            (ax1, f1_vals,  "Relaxed F1 by Ablation Variant",  "Relaxed F1"),
            (ax2, bs_vals,  "BERTScore F1 by Ablation Variant", "BERTScore F1"),
        ]:
            bars = ax.bar(
                x, vals, width=0.65,
                color=colors, hatch=hatches,
                edgecolor="white", linewidth=0.6, alpha=0.92, zorder=2,
            )
            self._zoom_ylim(ax, vals, lo_pad=0.25, hi_pad=0.30)
            self._annotate_bars(ax, bars, vals, fmt=".4f", fontsize=7.0)

            ax.set_xticks(x)
            ax.set_xticklabels([_display(l) for l in labels], rotation=30, ha="right", fontsize=7.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold", fontsize=9)

            # Annotate latency in muted text below bar top
            ylo, yhi = ax.get_ylim()
            for bar, lat in zip(bars, lat_vals):
                if lat > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        ylo + (yhi - ylo) * 0.03,
                        f"{lat:.2f} ms",
                        ha="center", va="bottom",
                        fontsize=6.5, color="#777777", style="italic",
                    )

        fig.suptitle("Ablation Study: Impact of Training Design Choices",
                     fontsize=10, fontweight="bold", y=1.01)

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Eval vs latency bubble plot ───────────────────────────────────────────

    def plot_eval_vs_latency(
        self,
        summary: List[Dict],
        output_filename: str = "eval_vs_latency_plot.pdf",
    ) -> str:
        """
        Bubble scatter plot: x=latency, y=Relaxed F1, bubble_size∝BERTScore F1.
        SRAS variants are drawn with solid markers; baselines with hollow markers.
        A Pareto frontier line connects efficient non-dominated variants.
        """
        plt = self._get_matplotlib()
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(5.5, 4.0))

        xs, ys, bs_list, names = [], [], [], []
        for e in summary:
            name = e.get("name", "?")
            f1   = e.get("relaxed_f1",    0.0)
            lat  = e.get("avg_latency_ms", 0.0)
            bs   = e.get("bertscore_f1",  0.0)
            xs.append(lat); ys.append(f1); bs_list.append(bs); names.append(name)

        bubble_scale = 800
        for x, y, bs, name in zip(xs, ys, bs_list, names):
            is_sras = name in _SRAS_TRAINED
            col = _color(name, _GRAY)
            sz  = max(bs * bubble_scale, 30)
            marker = "o" if is_sras else "D"
            ax.scatter(
                x, y, s=sz,
                color=col, marker=marker,
                alpha=0.85 if is_sras else 0.70,
                edgecolors="white" if is_sras else col,
                linewidths=0.8 if is_sras else 1.2,
                zorder=3,
            )
            ax.annotate(
                _display(name).replace("\n", " "),
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=6.5,
                color="#333333",
            )

        # Pareto frontier (maximise F1, minimise latency)
        pts = sorted(zip(xs, ys), key=lambda p: p[0])
        pareto = []
        best_y = -1.0
        for px, py in pts:
            if py > best_y:
                pareto.append((px, py))
                best_y = py
        if len(pareto) >= 2:
            px_vals, py_vals = zip(*pareto)
            ax.plot(
                px_vals, py_vals,
                linestyle="--", color="#555555", linewidth=0.9,
                label="Pareto frontier", zorder=2,
            )

        ax.set_xlabel("Average Inference Latency (ms)")
        ax.set_ylabel("Relaxed F1")
        ax.set_title("Accuracy–Latency Trade-off\n(bubble area ∝ BERTScore F1)", fontweight="bold")

        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_BLUE,
                   markersize=7, label="SRAS variants"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=_GRAY,
                   markersize=6, label="Baselines"),
            Line2D([0], [0], linestyle="--", color="#555555",
                   linewidth=0.9, label="Pareto frontier"),
        ]
        ax.legend(handles=legend_elems, loc="lower right",
                  framealpha=0.9, edgecolor="none")

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Compression comparison ────────────────────────────────────────────────

    def plot_compression_comparison(
        self,
        compression_results: List[Dict],
        output_filename: str = "compression_comparison.pdf",
    ) -> str:
        """Three-panel figure: latency, model-size, and sparsity vs params."""
        plt = self._get_matplotlib()

        if not compression_results:
            logger.warning("No compression results to plot.")
            return ""

        labels   = [r.get("label", f"v{i}") for i, r in enumerate(compression_results)]
        latencies = [r.get("latency_ms", r.get("p50_ms", 0.0)) for r in compression_results]
        sizes    = [r.get("model_size_mb", 0.0) for r in compression_results]
        sparsity = [r.get("sparsity", 0.0) for r in compression_results]
        params   = [r.get("num_params", 0) for r in compression_results]

        x = np.arange(len(labels))
        palette = [_BLUE, _ORANGE, _GREEN, _PURPLE, _GRAY]
        colors  = [palette[i % len(palette)] for i in range(len(labels))]

        fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5))

        for ax, ys, ylabel, title in [
            (axes[0], latencies, "Latency (ms)",     "Inference Latency"),
            (axes[1], sizes,     "Model Size (MB)",  "Model Size"),
            (axes[2], sparsity,  "Weight Sparsity",  "Weight Sparsity"),
        ]:
            bars = ax.bar(x, ys, color=colors, alpha=0.88, edgecolor="white", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold", fontsize=9)
            self._zoom_ylim(ax, ys, lo_pad=0.0, hi_pad=0.20)
            self._annotate_bars(ax, bars, ys, fmt=".3g", fontsize=6.5)

        # Overlay param count on sparsity panel
        ax3 = axes[2].twinx()
        ax3.plot(x, params, color=_PURPLE, marker="s", linewidth=1.5,
                 markersize=4, label="Num params")
        ax3.set_ylabel("# Parameters", color=_PURPLE, fontsize=8)
        ax3.tick_params(axis="y", labelcolor=_PURPLE, labelsize=7.5)
        ax3.spines["right"].set_visible(True)

        fig.suptitle("Compression Trade-off Analysis",
                     fontsize=10, fontweight="bold", y=1.02)

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Robustness sweep ──────────────────────────────────────────────────────

    def plot_robustness_sweep(
        self,
        robustness_results: Dict,
        output_filename: str = "robustness_sweep.pdf",
    ) -> str:
        """
        Three-panel line plot: noise / redundant / adversarial rate vs mean F1.
        Shaded ±1 std band from multi-trial data.
        """
        plt = self._get_matplotlib()

        sweep_configs = [
            ("noise",       "noise_rate",       "Noise Rate"),
            ("redundant",   "redundant_rate",   "Redundancy Rate"),
            ("adversarial", "adversarial_rate", "Adversarial Rate"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
        colors = [_BLUE, _ORANGE, _GREEN]

        for (key, x_key, xlabel), ax, col in zip(sweep_configs, axes, colors):
            data = robustness_results.get(key, [])
            if not data:
                ax.set_title(f"{xlabel}\n(no data)")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="#aaaaaa")
                continue

            xs   = np.array([d.get(x_key, 0.0) for d in data])
            ys   = np.array([d.get("mean_f1", 0.0) for d in data])
            trials = [d.get("trials", [d.get("mean_f1", 0.0)]) for d in data]

            # Confidence band from trials
            stds = np.array([np.std(t) if len(t) > 1 else 0.0 for t in trials])

            ax.fill_between(xs, ys - stds, ys + stds, color=col, alpha=0.15, zorder=1)
            ax.plot(xs, ys, marker="o", color=col, linewidth=1.8,
                    markersize=5, zorder=3, label="Mean F1 ± std")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean Relaxed F1")
            ax.set_title(f"Robustness: {xlabel}", fontweight="bold", fontsize=9)
            self._zoom_ylim(ax, list(ys), lo_pad=0.2, hi_pad=0.2)
            ax.legend(fontsize=7, framealpha=0.9, edgecolor="none")

        fig.suptitle("SRAS-PPO Robustness to Corpus Perturbations",
                     fontsize=10, fontweight="bold", y=1.02)

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Domain shift ──────────────────────────────────────────────────────────

    def plot_domain_shift(
        self,
        domain_shift_results: List[Dict],
        output_filename: str = "domain_shift.pdf",
    ) -> str:
        plt = self._get_matplotlib()

        if not domain_shift_results:
            logger.warning("No domain shift results to plot.")
            return ""

        categories = [d.get("category", f"cat_{i}") for i, d in enumerate(domain_shift_results)]
        f1s = np.array([d.get("mean_f1", 0.0) for d in domain_shift_results])
        stds = np.array([
            np.std(d.get("trials", [d.get("mean_f1", 0.0)])) for d in domain_shift_results
        ])
        x = np.arange(len(categories))

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        bars = ax.bar(
            x, f1s, width=0.6,
            color=_GREEN, alpha=0.88, edgecolor="white", linewidth=0.6,
            yerr=stds, capsize=3, error_kw={"linewidth": 0.9, "color": "#555555"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylabel("Mean Relaxed F1")
        ax.set_title("Domain Shift Robustness by Category", fontweight="bold")
        self._zoom_ylim(ax, list(f1s), lo_pad=0.2, hi_pad=0.25)
        self._annotate_bars(ax, bars, f1s, fmt=".4f", fontsize=7.0)

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Failure breakdown ─────────────────────────────────────────────────────

    def plot_failure_breakdown(
        self,
        failure_results: Dict,
        output_filename: str = "failure_breakdown.pdf",
    ) -> str:
        """
        Two-panel figure per model variant:
          Left: stacked bar of selector-fail vs generator-fail rates per question type.
          Right: mean F1 per question type.
        Supports both FailureAnalyzer output and SelectorEvaluator output.
        """
        plt = self._get_matplotlib()
        from matplotlib.patches import Patch

        per_qtype = failure_results.get("per_question_type", [])
        if not per_qtype:
            logger.warning("No per-question-type data in failure results.")
            return ""

        qtypes = [d["question_type"] for d in per_qtype]
        counts = [max(d.get("total", d.get("count", 1)), 1) for d in per_qtype]

        sel_rates, gen_rates, mean_f1s = [], [], []
        for d, cnt in zip(per_qtype, counts):
            # Selector failure rate
            if "selector_failure_rate" in d:
                sr = d["selector_failure_rate"]
            else:
                sr = d.get("failure_count", 0) / cnt
            sel_rates.append(sr)

            # Generator failure rate
            if "generator_failure_rate" in d:
                gr = d["generator_failure_rate"]
            else:
                mf1 = d.get("mean_f1", 0.0)
                gr = max(0.0, (1.0 - mf1) - sr)
            gen_rates.append(gr)

            mean_f1s.append(d.get("mean_f1", 0.0))

        sel_rates = np.array(sel_rates)
        gen_rates = np.array(gen_rates)
        mean_f1s  = np.array(mean_f1s)

        x = np.arange(len(qtypes))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

        # ── Left: stacked failure bars ────────────────────────────────────────
        bars_sel = ax1.bar(x, sel_rates, width=0.6, label="Selector failure",
                           color=_ORANGE, alpha=0.88, edgecolor="white", linewidth=0.6)
        bars_gen = ax1.bar(x, gen_rates, width=0.6, bottom=sel_rates,
                           label="Generator failure",
                           color=_BLUE, alpha=0.72, edgecolor="white", linewidth=0.6, hatch="//")

        ax1.set_xticks(x)
        ax1.set_xticklabels(qtypes, rotation=30, ha="right")
        ax1.set_ylabel("Failure Rate")
        ax1.set_title("Failure Attribution by Question Type", fontweight="bold", fontsize=9)
        ax1.set_ylim(0, min(1.02, max(sel_rates + gen_rates) * 1.30 + 0.05))
        ax1.legend(
            handles=[
                Patch(facecolor=_ORANGE, label="Selector failure"),
                Patch(facecolor=_BLUE, hatch="//", alpha=0.72, label="Generator failure"),
            ],
            loc="upper right", framealpha=0.9, edgecolor="none",
        )

        # Annotate counts above each stack
        for xi, (sr, gr, cnt) in enumerate(zip(sel_rates, gen_rates, counts)):
            total_rate = sr + gr
            ax1.text(xi, total_rate + 0.01, f"n={cnt}",
                     ha="center", va="bottom", fontsize=6.5, color="#555555")

        # ── Right: mean F1 per question type ─────────────────────────────────
        bars_f1 = ax2.bar(x, mean_f1s, width=0.6, color=_GREEN,
                          alpha=0.88, edgecolor="white", linewidth=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(qtypes, rotation=30, ha="right")
        ax2.set_ylabel("Mean Relaxed F1")
        ax2.set_title("Mean F1 by Question Type", fontweight="bold", fontsize=9)
        self._zoom_ylim(ax2, list(mean_f1s), lo_pad=0.25, hi_pad=0.25)
        self._annotate_bars(ax2, bars_f1, mean_f1s, fmt=".4f", fontsize=6.5)

        # Overall summary text in subtitle
        summary = failure_results.get("summary", {})
        overall_f1  = summary.get("overall_mean_f1", float(np.mean(mean_f1s)) if len(mean_f1s) else 0.0)
        sel_rate_ov = summary.get("selector_failure_rate", float(np.mean(sel_rates)) if len(sel_rates) else 0.0)
        gen_rate_ov = summary.get("generator_failure_rate", float(np.mean(gen_rates)) if len(gen_rates) else 0.0)
        fig.suptitle(
            f"Failure Analysis   |   Overall F1: {overall_f1:.4f}   "
            f"|   Selector fail: {sel_rate_ov:.3f}   "
            f"|   Generator fail: {gen_rate_ov:.3f}",
            fontsize=9, y=1.02,
        )

        out_path = os.path.join(self.output_dir, output_filename)
        return self._save(plt, fig, out_path)

    # ── Summary table builder ─────────────────────────────────────────────────

    def build_summary_table(
        self,
        eval_results: Dict[str, Dict],
        benchmark_results: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        latency_map: Dict[str, float] = {}
        if benchmark_results:
            for entry in benchmark_results:
                name = entry.get("variant", entry.get("name", ""))
                if name not in latency_map:
                    latency_map[name] = entry.get("avg_latency_ms", 0.0)

        summary: List[Dict] = []
        for variant, result in eval_results.items():
            if isinstance(result, dict) and "metrics" in result:
                metrics = result["metrics"]
            elif isinstance(result, dict):
                metrics = result
            else:
                metrics = {}

            summary.append({
                "name":           variant,
                "relaxed_f1":     metrics.get("relaxed_f1",    0.0),
                "bertscore_f1":   metrics.get("bertscore_f1",  0.0),
                "exact_match":    metrics.get("exact_match",   0.0),
                "avg_latency_ms": latency_map.get(variant,     0.0),
            })
        return summary
