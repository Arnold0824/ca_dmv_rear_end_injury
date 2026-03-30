from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PACKAGE_ROOT / "results" / "inferential"
DEEP_SHAP_DIR = PACKAGE_ROOT / "results" / "explainable_ml"
FIGURES_DIR = PACKAGE_ROOT / "manuscript_assets" / "figures"
TABLES_DIR = PACKAGE_ROOT / "manuscript_assets" / "tables"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4b4b4b",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_terms = pd.read_csv(ANALYSIS_DIR / "model_terms.csv")
    shap_metrics = pd.read_csv(DEEP_SHAP_DIR / "variant_metrics.csv")
    return model_terms, shap_metrics


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#29465b"):
    shadow = FancyBboxPatch(
        (xy[0] + 0.006, xy[1] - 0.01),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=0,
        edgecolor="none",
        facecolor="#edf1f5",
        transform=ax.transAxes,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.05,
        edgecolor="#ccd6de",
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    x0, y0 = xy
    accent = FancyBboxPatch(
        (x0, y0 + height - 0.028),
        width,
        0.028,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=0,
        edgecolor="none",
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(accent)
    ax.text(
        x0 + width / 2,
        y0 + height * 0.66,
        title,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color="#153243",
    )
    ax.text(
        x0 + width / 2,
        y0 + height * 0.33,
        body,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9.8,
        color="#20323f",
        linespacing=1.2,
    )


def add_arrow(ax, start, end, color="#61788a"):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.3,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def build_methods_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    ax.set_axis_off()

    blue = "#d8e9f3"
    teal = "#d8efe8"
    sand = "#f4ead2"
    rose = "#f6ddd8"

    add_box(
        ax,
        (0.04, 0.60),
        0.21,
        0.22,
        "PDF-derived archive",
        "832 parsed reports\nraw, normalized, and feature tables",
        blue,
    )
    add_box(
        ax,
        (0.30, 0.60),
        0.19,
        0.22,
        "Analysis sample",
        "340 reports\n109 injury cases",
        blue,
    )
    add_box(
        ax,
        (0.54, 0.60),
        0.19,
        0.22,
        "Weak-label check",
        "109 one-to-one overlaps\n0.92 agreement",
        sand,
    )
    add_box(
        ax,
        (0.78, 0.60),
        0.18,
        0.22,
        "Primary inference",
        "Binomial logit\nmean-centered interaction terms",
        teal,
    )

    add_box(
        ax,
        (0.20, 0.18),
        0.27,
        0.22,
        "Nested model sequence",
        "Baseline\nContext\nStopped-context\nStopped-context + operator",
        teal,
    )
    add_box(
        ax,
        (0.52, 0.18),
        0.22,
        0.22,
        "Robustness",
        "Stopped-only subgroup\n300 bootstrap replications\nLeave-one-operator-out",
        rose,
    )
    add_box(
        ax,
        (0.79, 0.18),
        0.17,
        0.22,
        "Secondary explanation",
        "LightGBM\n5-fold CV\nSHAP importance + dependence + local cases",
        rose,
    )

    add_arrow(ax, (0.25, 0.71), (0.30, 0.71))
    add_arrow(ax, (0.49, 0.71), (0.54, 0.71))
    add_arrow(ax, (0.73, 0.71), (0.78, 0.71))
    add_arrow(ax, (0.87, 0.60), (0.87, 0.41))
    add_arrow(ax, (0.61, 0.40), (0.61, 0.49))
    add_arrow(ax, (0.47, 0.29), (0.52, 0.29))
    add_arrow(ax, (0.74, 0.29), (0.79, 0.29))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure4_methods_workflow.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_matrix(ax, row_labels, col_labels, included, palette, letter):
    ax.set_xlim(0, len(col_labels))
    ax.set_ylim(0, len(row_labels))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(col_labels))])
    display_cols = [label.replace(" / ", "/\n").replace("Stopped cue", "Stopped\ncue").replace("Mechanism text", "Mechanism\ntext") for label in col_labels]
    ax.set_xticklabels(display_cols)
    ax.set_yticks([i + 0.5 for i in range(len(row_labels))])
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0, axis="x", labelsize=9.5)
    ax.tick_params(length=0, axis="y", labelsize=10)
    ax.text(0.0, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")
    for y, row in enumerate(row_labels):
        for x, col in enumerate(col_labels):
            key = (row, col)
            face = palette["on"] if included.get(key, False) else palette["off"]
            rect = Rectangle((x, y), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            if included.get(key, False):
                ax.text(x + 0.5, y + 0.52, "●", ha="center", va="center", fontsize=11, color="#173449")
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_model_design_figure() -> None:
    fig = plt.figure(figsize=(13.0, 7.0), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 0.95], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    inferential_rows = ["Baseline", "Context", "Stopped", "Stopped + operator"]
    inferential_cols = ["Rear-end", "Mode", "Interaction", "VRU", "Night", "Intersection", "Stopped cue", "Operators"]
    inferential_map = {
        ("Baseline", "Rear-end"): True,
        ("Baseline", "Mode"): True,
        ("Baseline", "Interaction"): True,
        ("Context", "Rear-end"): True,
        ("Context", "Mode"): True,
        ("Context", "Interaction"): True,
        ("Context", "VRU"): True,
        ("Context", "Night"): True,
        ("Context", "Intersection"): True,
        ("Stopped", "Rear-end"): True,
        ("Stopped", "Mode"): True,
        ("Stopped", "Interaction"): True,
        ("Stopped", "VRU"): True,
        ("Stopped", "Night"): True,
        ("Stopped", "Intersection"): True,
        ("Stopped", "Stopped cue"): True,
        ("Stopped + operator", "Rear-end"): True,
        ("Stopped + operator", "Mode"): True,
        ("Stopped + operator", "Interaction"): True,
        ("Stopped + operator", "VRU"): True,
        ("Stopped + operator", "Night"): True,
        ("Stopped + operator", "Intersection"): True,
        ("Stopped + operator", "Stopped cue"): True,
        ("Stopped + operator", "Operators"): True,
    }
    draw_matrix(
        ax1,
        inferential_rows,
        inferential_cols,
        inferential_map,
        palette={"on": "#a8c8da", "off": "#eef3f6"},
        letter="A",
    )

    ml_rows = ["Scene-only", "Scene + mechanism", "Scene + mechanism + operator"]
    ml_cols = ["Rear-end", "Mode", "Stopped", "VRU", "Intersection", "Lane / turn", "Signal / parking", "Mechanism text", "Operators"]
    ml_map = {
        ("Scene-only", "Rear-end"): True,
        ("Scene-only", "Mode"): True,
        ("Scene-only", "Stopped"): True,
        ("Scene-only", "VRU"): True,
        ("Scene-only", "Intersection"): True,
        ("Scene-only", "Lane / turn"): True,
        ("Scene-only", "Signal / parking"): True,
        ("Scene + mechanism", "Rear-end"): True,
        ("Scene + mechanism", "Mode"): True,
        ("Scene + mechanism", "Stopped"): True,
        ("Scene + mechanism", "VRU"): True,
        ("Scene + mechanism", "Intersection"): True,
        ("Scene + mechanism", "Lane / turn"): True,
        ("Scene + mechanism", "Signal / parking"): True,
        ("Scene + mechanism", "Mechanism text"): True,
        ("Scene + mechanism + operator", "Rear-end"): True,
        ("Scene + mechanism + operator", "Mode"): True,
        ("Scene + mechanism + operator", "Stopped"): True,
        ("Scene + mechanism + operator", "VRU"): True,
        ("Scene + mechanism + operator", "Intersection"): True,
        ("Scene + mechanism + operator", "Lane / turn"): True,
        ("Scene + mechanism + operator", "Signal / parking"): True,
        ("Scene + mechanism + operator", "Mechanism text"): True,
        ("Scene + mechanism + operator", "Operators"): True,
    }
    draw_matrix(
        ax2,
        ml_rows,
        ml_cols,
        ml_map,
        palette={"on": "#d8c5dd", "off": "#f4eff6"},
        letter="B",
    )

    fig.savefig(FIGURES_DIR / "Figure5_model_design.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_model_spec_table() -> None:
    rows = [
        ("Baseline", "Rear-end, autonomous mode, centered interaction", "Baseline interaction model without stopped-context or operator controls."),
        ("Context", "Baseline + VRU + night + intersection-like location", "Adds scene-level controls that commonly explain severity heterogeneity."),
        ("Stopped-context", "Context + stopped-traffic narrative cue", "Tests whether queue-like stopping explains the interaction."),
        ("Stopped-context + operator", "Stopped-context + top-3 manufacturer controls", "Checks whether the signal survives operator mix."),
    ]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Nested inferential model specifications.}",
        "\\label{tab:model_specs}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{p{2.4cm}p{5.0cm}p{5.2cm}}",
        "\\toprule",
        "Specification & Included blocks & Analytical purpose \\\\",
        "\\midrule",
    ]
    for spec, blocks, purpose in rows:
        lines.append(f"{spec} & {blocks} & {purpose} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table4_model_specs.tex").write_text("\n".join(lines) + "\n")


def write_robustness_table() -> None:
    rows = [
        ("Stopped-only subgroup model", "n = 220 and n = 120 by subgroup", "Checks whether the interaction is concentrated in stopped traffic rather than spread uniformly."),
        ("Bootstrap interaction stability", "300 replications", "Quantifies how often the centered rear-end × autonomous interaction remains positive under resampling."),
        ("Leave-one-operator-out", "Top 6 operators", "Assesses sensitivity of the main interaction to operator mix and deployment style."),
        ("Scene-only LightGBM", "5-fold stratified CV", "Provides a non-linear benchmark without operator identifiers."),
        ("Scene + mechanism LightGBM", "5-fold stratified CV", "Tests whether finer regex-derived stop and rear-end mechanism cues add explanatory value."),
        ("Scene + mechanism + operator LightGBM", "5-fold stratified CV", "Tests whether operator features still dominate once richer text mechanisms are introduced."),
        ("Deep SHAP diagnostics", "Dependence summaries and local cases", "Audits whether global importance, feature dependence, and case-level explanations tell a coherent mechanism story."),
    ]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Robustness and explainability protocol.}",
        "\\label{tab:robustness_protocol}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{p{3.1cm}p{2.6cm}p{6.4cm}}",
        "\\toprule",
        "Component & Scope & Purpose \\\\",
        "\\midrule",
    ]
    for comp, scope, purpose in rows:
        lines.append(f"{comp} & {scope} & {purpose} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table5_robustness_protocol.tex").write_text("\n".join(lines) + "\n")


def write_lightgbm_table(shap_metrics: pd.DataFrame) -> None:
    metrics = shap_metrics.pivot(index="variant", columns="metric", values="value").reset_index()
    rows = []
    for _, row in metrics.iterrows():
        variant = row["variant"]
        if variant == "scene_only":
            blocks = "Scene, mode, rear-end, stopped, maneuver, signal, and VRU features"
            label = "Scene-only"
        elif variant == "scene_plus_mechanism":
            blocks = "Scene-only block plus regex-derived stop, signal, braking, yielding, and rear-strike mechanism cues"
            label = "Scene + mechanism"
        else:
            blocks = "Scene + mechanism block plus Cruise, Waymo, and Zoox dummies"
            label = "Scene + mechanism + operator"
        rows.append(
            (
                label,
                blocks,
                "5-fold stratified CV; SHAP importance, pairwise interactions, dependence summaries, and local cases",
            )
        )

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Secondary LightGBM variants used in the explanation layer.}",
        "\\label{tab:lightgbm_config}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{p{2.8cm}p{5.2cm}p{5.2cm}}",
        "\\toprule",
        "Variant & Feature blocks & Validation and explanation outputs \\\\",
        "\\midrule",
    ]
    for variant, blocks, outputs in rows:
        lines.append(f"{variant} & {blocks} & {outputs} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table6_lightgbm_config.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    _, shap_metrics = load_outputs()
    build_model_design_figure()
    write_model_spec_table()
    write_robustness_table()
    write_lightgbm_table(shap_metrics)


if __name__ == "__main__":
    main()
