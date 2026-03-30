from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PACKAGE_ROOT / "results" / "explainable_ml"
FIGURES_DIR = ANALYSIS_DIR / "figures"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
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


def add_panel_letter(ax, letter: str) -> None:
    ax.text(0.0, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")


def pretty_feature(name: str) -> str:
    label_map = {
        "manu_Cruise": "Cruise",
        "manu_Zoox": "Zoox",
        "manu_Waymo": "Waymo",
        "location_is_intersection_like": "Intersection-like location",
        "narrative_mentions_stopped": "Stopped cue",
        "collision_type__rear_end": "Rear-end",
        "mech_stop_context_expanded": "Expanded stop context",
        "vru_any": "VRU involvement",
        "narrative_mentions_entered_lane": "Entered-lane cue",
        "is_night": "Night",
        "narrative_mentions_red_light": "Red-light cue",
        "mech_signal_wait": "Signal wait",
        "mech_yielding": "Yielding",
        "mech_braking": "Braking",
        "mech_stop_in_lane": "Stopped in lane",
        "mode_binary_autonomous": "Autonomous mode",
        "narrative_mentions_right_turn": "Right-turn cue",
        "narrative_mentions_left_turn": "Left-turn cue",
    }
    return label_map.get(name, name.replace("_", " "))


def load() -> dict[str, pd.DataFrame]:
    files = [
        "variant_metrics.csv",
        "feature_catalog.csv",
        "shap_importance.csv",
        "shap_interactions.csv",
        "interaction_cell_summary.csv",
        "dependence_summary.csv",
        "local_case_explanations.csv",
        "local_case_selection.csv",
    ]
    loaded: dict[str, pd.DataFrame] = {}
    for file_name in files:
        path = ANALYSIS_DIR / file_name
        if path.exists():
            loaded[file_name] = pd.read_csv(path)
    return loaded


def build_performance_and_importance(metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True)

    pivot = metrics.pivot(index="variant", columns="metric", values="value").reset_index()
    order = ["scene_only", "scene_plus_mechanism", "scene_plus_mechanism_operator"]
    labels = ["Scene only", "Scene + mechanism", "Scene + mechanism + operator"]
    pivot["variant"] = pd.Categorical(pivot["variant"], categories=order, ordered=True)
    pivot = pivot.sort_values("variant")
    colors = ["#7aa6c2", "#95b38b", "#b78fb8"]
    axes[0].bar(labels, pivot["cv_auc_mean"], color=colors, width=0.62, alpha=0.92)
    axes[0].errorbar(
        labels,
        pivot["cv_auc_mean"],
        yerr=pivot["cv_auc_sd"],
        fmt="none",
        ecolor="#465c6a",
        elinewidth=1.4,
        capsize=3,
        capthick=1.4,
    )
    for i, value in enumerate(pivot["cv_auc_mean"]):
        axes[0].text(i, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("Cross-validated AUC")
    axes[0].set_ylim(0.45, max(0.75, float(pivot["cv_auc_mean"].max()) + 0.06))
    axes[0].tick_params(axis="x", rotation=14)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    add_panel_letter(axes[0], "A")

    top = (
        importance[importance["variant"] == "scene_plus_mechanism_operator"]
        .head(8)
        .iloc[::-1]
        .copy()
    )
    pretty = top["feature"].map(pretty_feature)
    axes[1].hlines(pretty, xmin=0, xmax=top["mean_abs_shap"], color="#d9a087", linewidth=2.0, alpha=0.85)
    axes[1].scatter(top["mean_abs_shap"], pretty, s=80, color="#b55b52", zorder=3)
    axes[1].set_xlabel("Mean |SHAP|")
    axes[1].grid(axis="x", linestyle="--", alpha=0.25)
    add_panel_letter(axes[1], "B")

    fig.savefig(FIGURES_DIR / "deep_shap_performance_importance.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_dependence_summary(dependence: pd.DataFrame) -> None:
    selected = [
        "collision_type__rear_end",
        "narrative_mentions_stopped",
        "mech_stop_context_expanded",
        "mech_signal_wait",
        "mech_yielding",
        "manu_Cruise",
    ]
    label_map = {
        "collision_type__rear_end": "Rear-end",
        "narrative_mentions_stopped": "Stopped cue",
        "mech_stop_context_expanded": "Expanded stop context",
        "mech_signal_wait": "Signal wait",
        "mech_yielding": "Yielding",
        "manu_Cruise": "Cruise",
    }
    sub = dependence[dependence["variant"] == "scene_plus_mechanism_operator"].copy()
    sub = sub[sub["feature"].isin(selected)].copy()
    sub["feature"] = pd.Categorical(sub["feature"], categories=selected, ordered=True)
    sub = sub.sort_values(["feature", "feature_value"])

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    y = np.arange(len(selected))
    offsets = {0: -0.14, 1: 0.14}
    color_map = {0: "#9ebed2", 1: "#d08159"}
    for feature_value in [0, 1]:
        part = sub[sub["feature_value"] == feature_value].copy()
        values = [part.loc[part["feature"] == feat, "mean_shap"].iloc[0] if (part["feature"] == feat).any() else 0 for feat in selected]
        label = "Present" if feature_value == 1 else "Absent"
        ax.scatter(values, y + offsets[feature_value], s=80, color=color_map[feature_value], label=label)
        for value, yy in zip(values, y + offsets[feature_value]):
            ax.plot([0, value], [yy, yy], color=color_map[feature_value], alpha=0.35, linewidth=1.6)
    ax.axvline(0.0, linestyle="--", color="#7f7f7f", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map[f] for f in selected])
    ax.set_xlabel("Mean SHAP contribution to injury prediction")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIGURES_DIR / "deep_shap_dependence_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_local_cases(case_explanations: pd.DataFrame) -> None:
    if case_explanations.empty:
        return

    groups = case_explanations["case_group"].drop_duplicates().tolist()[:4]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    axes = axes.flatten()
    color_pos = "#d08159"
    color_neg = "#9ebed2"

    for ax, group, letter in zip(axes, groups, ["A", "B", "C", "D"]):
        sub = case_explanations[case_explanations["case_group"] == group].copy()
        sub = sub.sort_values("shap_value")
        sub["feature_pretty"] = sub["feature"].map(pretty_feature)
        colors = [color_pos if v > 0 else color_neg for v in sub["shap_value"]]
        ax.barh(sub["feature_pretty"], sub["shap_value"], color=colors)
        ax.axvline(0.0, linestyle="--", color="#7f7f7f", linewidth=1.0)
        meta = sub.iloc[0]
        short_group = (
            group.replace("av", "AV")
            .replace("conv", "Conventional")
            .replace("_", " ")
        )
        ax.text(
            0.01,
            0.02,
            f"{short_group}\nPred={meta['pred_prob']:.2f}, Injury={int(meta['injury_text_signal'])}",
            transform=ax.transAxes,
            fontsize=9.5,
            va="bottom",
            color="#425c6f",
        )
        add_panel_letter(ax, letter)
        ax.grid(axis="x", linestyle="--", alpha=0.2)

    for ax in axes[len(groups):]:
        ax.axis("off")

    fig.savefig(FIGURES_DIR / "deep_shap_local_cases.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    data = load()
    if not data:
        raise SystemExit("No tree-explanation outputs found. Run scripts/modeling/run_tree_explanations.py first.")

    build_performance_and_importance(data["variant_metrics.csv"], data["shap_importance.csv"])
    build_dependence_summary(data["dependence_summary.csv"])
    build_local_cases(data.get("local_case_explanations.csv", pd.DataFrame()))
    print("Saved figures to", FIGURES_DIR)


if __name__ == "__main__":
    main()
