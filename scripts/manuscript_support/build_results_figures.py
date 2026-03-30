from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


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


def load_outputs() -> dict[str, pd.DataFrame]:
    names = [
        "key_inference_terms.csv",
        "logistic_model_terms.csv",
        "group_rates_by_mode_and_rear_end.csv",
        "group_rates_by_stop_context.csv",
        "stopped_context_subgroup_models.csv",
        "operator_leave_one_out.csv",
        "bootstrap_interaction_coefficients.csv",
        "../explainable_ml/lightgbm_variant_metrics.csv",
        "../explainable_ml/shap_importance.csv",
        "../explainable_ml/shap_interactions.csv",
        "../explainable_ml/lightgbm_variant_metrics.csv",
        "../explainable_ml/dependence_summary.csv",
        "../explainable_ml/local_case_selection.csv",
        "../explainable_ml/feature_catalog.csv",
        "../explainable_ml/interaction_cell_summary.csv",
    ]
    loaded: dict[str, pd.DataFrame] = {}
    for name in names:
        path = (ANALYSIS_DIR / name).resolve()
        loaded[name] = pd.read_csv(path)
    return loaded


def add_panel_letter(ax, letter: str) -> None:
    ax.text(0.0, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")


def build_overall_rates_figure(overall: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    color_map = {0: "#9ebed2", 1: "#d08159"}
    label_map = {0: "Non-rear-end", 1: "Rear-end"}
    x = np.arange(2)
    width = 0.34
    mode_order = ["autonomous", "conventional"]
    mode_labels = ["Autonomous", "Conventional"]

    for idx, rear_end in enumerate([0, 1]):
        vals = [
            overall.loc[(overall["mode_resolved"] == mode) & (overall["rear_end"] == rear_end), "injury_rate"].iloc[0]
            for mode in mode_order
        ]
        bars = ax.bar(x + (idx - 0.5) * width, vals, width=width, color=color_map[rear_end], label=label_map[rear_end])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("Injury rate")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.52, 1.10))
    fig.savefig(FIGURES_DIR / "Figure6_overall_injury_rates.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_stopped_split_figure(stopped: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.2, 5.6), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0.18)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    color_map = {0: "#9ebed2", 1: "#d08159"}
    label_map = {0: "Non-rear-end", 1: "Rear-end"}
    mode_order = ["autonomous", "conventional"]
    mode_labels = ["Autonomous", "Conventional"]
    x = np.arange(2)
    width = 0.34

    for ax, stopped_value, letter in zip(axes, [0, 1], ["A", "B"]):
        subset = stopped[stopped["narrative_mentions_stopped"] == stopped_value]
        add_panel_letter(ax, letter)
        for idx, rear_end in enumerate([0, 1]):
            vals = [
                subset.loc[(subset["mode_resolved"] == mode) & (subset["rear_end"] == rear_end), "injury_rate"].iloc[0]
                for mode in mode_order
            ]
            bars = ax.bar(x + (idx - 0.5) * width, vals, width=width, color=color_map[rear_end], label=label_map[rear_end])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.013, f"{val:.1%}", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        ax.set_ylim(0, 0.58)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if stopped_value == 0:
            ax.set_ylabel("Injury rate")
        ax.text(0.02, 0.95, "Stopped cue = 0" if stopped_value == 0 else "Stopped cue = 1", transform=ax.transAxes, va="top", fontsize=10.5, color="#425c6f")
    axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(FIGURES_DIR / "Figure7_stopped_split_rates.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_logit_results_figure(focal: pd.DataFrame, model_terms: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.6, 7.0), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 1.2], wspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    interaction = focal[focal["term"] == "rear_mode_int"].copy()
    model_order = ["baseline", "context", "stopped_context", "stopped_context_manufacturer"]
    interaction["model_name"] = pd.Categorical(interaction["model_name"], categories=model_order, ordered=True)
    interaction = interaction.sort_values("model_name")
    y = np.arange(len(interaction))
    ax1.errorbar(
        interaction["or"],
        y,
        xerr=[interaction["or"] - interaction["ci_low"], interaction["ci_high"] - interaction["or"]],
        fmt="o",
        color="#2b5876",
        ecolor="#7fa6bd",
        capsize=3,
        linewidth=1.5,
    )
    ax1.axvline(1.0, linestyle="--", color="#7f7f7f", linewidth=1.0)
    ax1.set_yticks(y)
    ax1.set_yticklabels(["Baseline", "Context", "Stopped", "Stopped + operator"])
    ax1.set_xlabel("Odds ratio for rear-end × autonomous")
    ax1.set_xscale("log")
    ax1.grid(axis="x", linestyle="--", alpha=0.25)
    add_panel_letter(ax1, "A")

    main = model_terms[model_terms["model_name"] == "stopped_context_manufacturer"].copy()
    keep_terms = [
        "rear_end_c",
        "mode_autonomous_c",
        "rear_mode_int",
        "narrative_mentions_stopped",
        "location_is_intersection_like",
        "vru_any",
        "is_night",
        "C(manufacturer_top3)[T.Waymo]",
        "C(manufacturer_top3)[T.Zoox]",
        "C(manufacturer_top3)[T.Other]",
    ]
    label_map = {
        "rear_end_c": "Rear-end",
        "mode_autonomous_c": "Autonomous mode",
        "rear_mode_int": "Rear-end × autonomous",
        "narrative_mentions_stopped": "Stopped cue",
        "location_is_intersection_like": "Intersection-like location",
        "vru_any": "VRU involvement",
        "is_night": "Night",
        "C(manufacturer_top3)[T.Waymo]": "Waymo",
        "C(manufacturer_top3)[T.Zoox]": "Zoox",
        "C(manufacturer_top3)[T.Other]": "Other operators",
    }
    main = main[main["term"].isin(keep_terms)].copy()
    main["label"] = main["term"].map(label_map)
    main["label"] = pd.Categorical(main["label"], categories=[label_map[t] for t in keep_terms][::-1], ordered=True)
    main = main.sort_values("label")
    y2 = np.arange(len(main))
    ax2.errorbar(
        main["or"],
        y2,
        xerr=[main["or"] - main["ci_low"], main["ci_high"] - main["or"]],
        fmt="o",
        color="#8a4f3d",
        ecolor="#c99b8a",
        capsize=3,
        linewidth=1.5,
    )
    ax2.axvline(1.0, linestyle="--", color="#7f7f7f", linewidth=1.0)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(main["label"])
    ax2.set_xlabel("Odds ratio in main model")
    ax2.set_xscale("log")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)
    add_panel_letter(ax2, "B")

    fig.savefig(FIGURES_DIR / "Figure8_logit_results.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_robustness_figure(bootstrap: pd.DataFrame, loo: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.4, 5.8), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    q025, q50, q975 = bootstrap["rear_mode_int"].quantile([0.025, 0.5, 0.975]).tolist()
    ax1.hist(bootstrap["rear_mode_int"], bins=24, color="#a8c8da", edgecolor="white")
    ax1.axvline(0.0, linestyle="--", color="#7f7f7f")
    ax1.axvline(q50, color="#2b5876", linewidth=1.6)
    ax1.axvspan(q025, q975, color="#dce9f1", alpha=0.8)
    ax1.set_xlabel("Bootstrap coefficient for rear-end × autonomous")
    ax1.set_ylabel("Replications")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    add_panel_letter(ax1, "A")

    loo = loo.copy()
    loo["ci_low"] = loo["rear_mode_int_coef"] - 1.96 * loo["rear_mode_int_se"]
    loo["ci_high"] = loo["rear_mode_int_coef"] + 1.96 * loo["rear_mode_int_se"]
    loo = loo.sort_values("rear_mode_int_coef")
    y = np.arange(len(loo))
    ax2.errorbar(
        loo["rear_mode_int_coef"],
        y,
        xerr=[loo["rear_mode_int_coef"] - loo["ci_low"], loo["ci_high"] - loo["rear_mode_int_coef"]],
        fmt="o",
        color="#8a4f3d",
        ecolor="#c99b8a",
        capsize=3,
    )
    ax2.axvline(0.0, linestyle="--", color="#7f7f7f")
    ax2.set_yticks(y)
    ax2.set_yticklabels(loo["dropped_manufacturer"])
    ax2.set_xlabel("Rear-end × autonomous coefficient")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)
    add_panel_letter(ax2, "B")

    fig.savefig(FIGURES_DIR / "Figure9_robustness.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def copy_deep_shap_figures() -> None:
    mapping = {
        DEEP_SHAP_DIR / "figures" / "deep_shap_performance_importance.png": FIGURES_DIR / "Figure10_shap_performance_importance.png",
        DEEP_SHAP_DIR / "figures" / "deep_shap_dependence_summary.png": FIGURES_DIR / "Figure11_shap_dependence.png",
        DEEP_SHAP_DIR / "figures" / "deep_shap_local_cases.png": FIGURES_DIR / "Figure12_shap_local_cases.png",
    }
    for src, dst in mapping.items():
        copyfile(src, dst)


def write_model_comparison_table(focal: pd.DataFrame) -> None:
    pivot = focal[focal["term"] == "rear_mode_int"][["model_name", "coef", "se", "z", "p_value", "or", "ci_low", "ci_high", "aic", "max_vif"]].copy()
    name_map = {
        "baseline": "Baseline",
        "context": "Context",
        "stopped_context": "Stopped-context",
        "stopped_context_manufacturer": "Stopped-context + operator",
    }
    pivot["Specification"] = pivot["model_name"].map(name_map)
    pivot["Interaction OR (95\\% CI)"] = pivot.apply(lambda r: f"{r['or']:.2f} ({r['ci_low']:.2f}, {r['ci_high']:.2f})", axis=1)
    pivot["p"] = pivot["p_value"].map(lambda v: f"{v:.4f}")
    pivot["AIC"] = pivot["aic"].map(lambda v: f"{v:.1f}")
    pivot["Max VIF"] = pivot["max_vif"].map(lambda v: f"{v:.2f}")
    rows = pivot[["Specification", "Interaction OR (95\\% CI)", "p", "AIC", "Max VIF"]].to_dict("records")

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Model comparison for the focal rear-end $\\times$ autonomous interaction.}",
        "\\label{tab:model_comparison}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{p{3.0cm}p{4.5cm}rrr}",
        "\\toprule",
        "Specification & Interaction OR (95\\% CI) & $p$ & AIC & Max VIF \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(f"{row['Specification']} & {row['Interaction OR (95\\% CI)']} & {row['p']} & {row['AIC']} & {row['Max VIF']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table7_model_comparison.tex").write_text("\n".join(lines) + "\n")


def write_main_model_table(model_terms: pd.DataFrame) -> None:
    main = model_terms[model_terms["model_name"] == "stopped_context_manufacturer"].copy()
    keep_terms = [
        "rear_end_c",
        "mode_autonomous_c",
        "rear_mode_int",
        "narrative_mentions_stopped",
        "location_is_intersection_like",
        "vru_any",
        "is_night",
        "C(manufacturer_top3)[T.Waymo]",
        "C(manufacturer_top3)[T.Zoox]",
        "C(manufacturer_top3)[T.Other]",
    ]
    label_map = {
        "rear_end_c": "Rear-end",
        "mode_autonomous_c": "Autonomous mode",
        "rear_mode_int": "Rear-end × autonomous",
        "narrative_mentions_stopped": "Stopped cue",
        "location_is_intersection_like": "Intersection-like location",
        "vru_any": "VRU involvement",
        "is_night": "Night",
        "C(manufacturer_top3)[T.Waymo]": "Waymo",
        "C(manufacturer_top3)[T.Zoox]": "Zoox",
        "C(manufacturer_top3)[T.Other]": "Other operators",
    }
    main = main[main["term"].isin(keep_terms)].copy()
    main["Term"] = main["term"].map(label_map)
    main["OR (95\\% CI)"] = main.apply(lambda r: f"{r['or']:.2f} ({r['ci_low']:.2f}, {r['ci_high']:.2f})", axis=1)
    main["p"] = main["p_value"].map(lambda v: f"{v:.4f}")
    main["Coef."] = main["coef"].map(lambda v: f"{v:.3f}")
    main["SE"] = main["se"].map(lambda v: f"{v:.3f}")
    rows = main[["Term", "Coef.", "SE", "OR (95\\% CI)", "p"]].to_dict("records")

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Main stopped-context plus operator model.}",
        "\\label{tab:main_model}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{p{3.5cm}rrp{4.3cm}r}",
        "\\toprule",
        "Term & Coef. & SE & OR (95\\% CI) & $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(f"{row['Term']} & {row['Coef.']} & {row['SE']} & {row['OR (95\\% CI)']} & {row['p']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table8_main_model.tex").write_text("\n".join(lines) + "\n")


def write_robustness_table(stopped_models: pd.DataFrame, loo: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    non_stopped = stopped_models[(stopped_models["stopped_value"] == 0) & (stopped_models["term"] == "rear_mode_int")].iloc[0]
    stopped = stopped_models[(stopped_models["stopped_value"] == 1) & (stopped_models["term"] == "rear_mode_int")].iloc[0]
    q025, q50, q975 = bootstrap["rear_mode_int"].quantile([0.025, 0.5, 0.975]).tolist()
    share_positive = (bootstrap["rear_mode_int"] > 0).mean()
    cruise = loo[loo["dropped_manufacturer"] == "Cruise"].iloc[0]
    others = loo[loo["dropped_manufacturer"] != "Cruise"]["rear_mode_int_p"]

    rows = [
        ("Stopped subgroup only", f"{stopped['coef']:.3f}", f"{stopped['p_value']:.4f}", "Interaction significant in stopped-traffic subset."),
        ("Non-stopped subgroup only", f"{non_stopped['coef']:.3f}", f"{non_stopped['p_value']:.4f}", "Interaction not significant outside stopped context."),
        ("Bootstrap 2.5\\% / 50\\% / 97.5\\%", f"{q025:.3f} / {q50:.3f} / {q975:.3f}", f"{share_positive:.3f}", "Share of positive bootstrap coefficients."),
        ("Leave out Cruise", f"{cruise['rear_mode_int_coef']:.3f}", f"{cruise['rear_mode_int_p']:.4f}", "Largest weakening of the focal interaction."),
        ("Leave out other major operators", f"{loo[loo['dropped_manufacturer'] != 'Cruise']['rear_mode_int_coef'].median():.3f}", f"{others.min():.4f}--{others.max():.4f}", "Interaction usually remains at or near significance."),
    ]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Robustness summary for the focal interaction.}",
        "\\label{tab:robustness_results}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{p{3.8cm}p{2.5cm}p{2.1cm}p{4.8cm}}",
        "\\toprule",
        "Check & Statistic & $p$ or share & Interpretation \\\\",
        "\\midrule",
    ]
    for name, stat, pv, interp in rows:
        lines.append(f"{name} & {stat} & {pv} & {interp} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table9_robustness_results.tex").write_text("\n".join(lines) + "\n")


def write_shap_table(shap_metrics: pd.DataFrame, interaction_cells: pd.DataFrame) -> None:
    metrics = shap_metrics.pivot(index="variant", columns="metric", values="value").reset_index()
    inter = interaction_cells.copy()

    def get_pair(variant: str, left: str, right: str) -> float:
        sub = inter[
            (inter["variant"] == variant)
            & (inter["feature_left"] == left)
            & (inter["feature_right"] == right)
            & (inter["left_value"] == 1)
            & (inter["right_value"] == 1)
        ]
        return float(sub["mean_abs_interaction_shap"].iloc[0]) if not sub.empty else float("nan")

    def fmt(value: float) -> str:
        return "--" if pd.isna(value) else f"{value:.3f}"

    rows = []
    label_map = {
        "scene_only": "Scene-only",
        "scene_plus_mechanism": "Scene + mechanism",
        "scene_plus_mechanism_operator": "Scene + mechanism + operator",
    }
    for _, row in metrics.iterrows():
        variant = row["variant"]
        rows.append(
            (
                label_map.get(variant, variant),
                f"{row['cv_auc_mean']:.3f}",
                f"{row['cv_accuracy_mean']:.3f}",
                fmt(get_pair(variant, 'collision_type__rear_end', 'mode_binary_autonomous')),
                fmt(get_pair(variant, 'collision_type__rear_end', 'narrative_mentions_stopped')),
                fmt(get_pair(variant, 'collision_type__rear_end', 'mech_stop_context_expanded')),
            )
        )

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{SHAP analysis: model performance and selected 2$\\times$2 interaction-cell magnitudes.}",
        "\\label{tab:shap_results}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{p{2.8cm}ccccc}",
        "\\toprule",
        "Variant & CV AUC & CV acc. & \\shortstack[c]{Rear-end $\\times$\\\\ autonomous} & \\shortstack[c]{Rear-end $\\times$\\\\ stopped cue} & \\shortstack[c]{Rear-end $\\times$\\\\ expanded stop} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table10_shap_results.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    data = load_outputs()

    build_overall_rates_figure(data["group_rates_by_mode_and_rear_end.csv"])
    build_stopped_split_figure(data["group_rates_by_stop_context.csv"])
    build_logit_results_figure(data["key_inference_terms.csv"], data["logistic_model_terms.csv"])
    build_robustness_figure(data["bootstrap_interaction_coefficients.csv"], data["operator_leave_one_out.csv"])
    copy_deep_shap_figures()

    write_model_comparison_table(data["key_inference_terms.csv"])
    write_main_model_table(data["logistic_model_terms.csv"])
    write_robustness_table(data["stopped_context_subgroup_models.csv"], data["operator_leave_one_out.csv"], data["bootstrap_interaction_coefficients.csv"])
    write_shap_table(data["../explainable_ml/lightgbm_variant_metrics.csv"], data["../explainable_ml/interaction_cell_summary.csv"])


if __name__ == "__main__":
    main()
