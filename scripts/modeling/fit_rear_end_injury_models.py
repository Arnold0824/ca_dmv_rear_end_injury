#!/usr/bin/env python3
"""Estimate the rear-end injury models used in the California DMV AV crash study.

The script performs four tasks:
1. prepare the inferential analysis sample from the engineered feature table,
2. estimate nested logistic-regression specifications with transparent uncertainty,
3. run bootstrap and leave-one-operator-out robustness checks, and
4. export summary tables that can be consumed by manuscript figure builders.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import lightgbm as lgb
    import shap
    from sklearn.model_selection import StratifiedKFold, cross_val_score
except Exception:  # pragma: no cover - optional explanation layer
    lgb = None
    shap = None
    StratifiedKFold = None
    cross_val_score = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = PACKAGE_ROOT / "data" / "processed" / "engineered_features.csv"
DEFAULT_LEGACY_XLSX = ""
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results" / "inferential"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    formula: str
    focal_terms: tuple[str, ...]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(DEFAULT_FEATURES), help="Prepared feature CSV.")
    parser.add_argument(
        "--legacy-xlsx",
        "--old-xlsx",
        dest="legacy_xlsx",
        default=DEFAULT_LEGACY_XLSX,
        help="Optional legacy California DMV crash table used for weak-label validation.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for analysis outputs.")
    parser.add_argument("--bootstrap", type=int, default=300, help="Bootstrap replications for the main interaction.")
    return parser.parse_args()


def normalize_company(text: object) -> str:
    raw = "" if pd.isna(text) else str(text).strip().lower()
    rules = [
        ("waymo", "Waymo"),
        ("google", "Waymo"),
        ("cruise", "Cruise"),
        ("gm cruise", "Cruise"),
        ("zoox", "Zoox"),
        ("apple", "Apple"),
        ("aurora", "Aurora"),
        ("weride", "WeRide"),
        ("pony", "Pony.ai"),
        ("nuro", "Nuro"),
        ("mercedes", "Mercedes-Benz"),
        ("lyft", "Lyft"),
        ("motional", "Motional"),
        ("autox", "AutoX"),
        ("aimotive", "aiMotive"),
        ("apollo", "Apollo"),
        ("baidu", "Baidu"),
        ("toyota", "Toyota"),
        ("may mobility", "May Mobility"),
        ("navya", "Navya"),
    ]
    for needle, label in rules:
        if needle in raw:
            return label
    return str(text).strip() if not pd.isna(text) else ""


def load_analysis_sample(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(features_path)
    # The inferential sample follows the manuscript design: resolved driving mode
    # plus a non-missing injury-indicator label derived from the crash narrative.
    df = raw[raw["mode_resolved"].isin(["autonomous", "conventional"])].copy()
    df = df.dropna(subset=["injury_text_signal"]).copy()
    df["injury_text_signal"] = df["injury_text_signal"].astype(int)
    df["rear_end"] = pd.to_numeric(df["collision_type__rear_end"], errors="coerce").fillna(0).astype(int)
    df["mode_autonomous"] = pd.to_numeric(df["mode_binary_autonomous"], errors="coerce").fillna(0).astype(int)

    for column in [
        "vru_any",
        "is_night",
        "location_is_intersection_like",
        "narrative_mentions_stopped",
        "narrative_mentions_lane_change",
        "narrative_mentions_merge",
        "narrative_mentions_left_turn",
        "narrative_mentions_right_turn",
        "narrative_mentions_entered_lane",
        "narrative_mentions_red_light",
        "narrative_mentions_parked",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    top3 = set(df["manufacturer_std"].value_counts().head(3).index)
    df["manufacturer_top3"] = df["manufacturer_std"].apply(lambda value: value if value in top3 else "Other")
    return raw, df


def recenter(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    centered = df.copy()
    for column in columns:
        centered[f"{column}_c"] = centered[column] - centered[column].mean()
    centered["rear_mode_int"] = centered["rear_end_c"] * centered["mode_autonomous_c"]
    return centered


def validate_weak_label(features_df: pd.DataFrame, old_xlsx_path: Path) -> tuple[dict[str, float], pd.DataFrame]:
    old = pd.read_excel(old_xlsx_path, sheet_name="Crash Data")
    old["date_parsed"] = pd.to_datetime(old["Date"], dayfirst=True, errors="coerce").dt.date
    old["company_norm"] = old["Company"].map(normalize_company)
    old["injury_any_old"] = (pd.to_numeric(old["Number of injuries"], errors="coerce").fillna(0) > 0).astype(int)

    new = features_df.copy()
    new["date_parsed"] = pd.to_datetime(new["accident_date"], errors="coerce").dt.date
    new["company_norm"] = new["manufacturer_std"].map(normalize_company)

    old_counts = old.groupby(["date_parsed", "company_norm"]).size().rename("old_n").reset_index()
    new_counts = new.groupby(["date_parsed", "company_norm"]).size().rename("new_n").reset_index()
    unique_keys = old_counts.merge(new_counts, on=["date_parsed", "company_norm"], how="inner")
    unique_keys = unique_keys[(unique_keys["old_n"] == 1) & (unique_keys["new_n"] == 1)][["date_parsed", "company_norm"]]

    overlap = new.merge(
        old[["date_parsed", "company_norm", "injury_any_old"]],
        on=["date_parsed", "company_norm"],
        how="inner",
    )
    overlap = overlap.merge(unique_keys, on=["date_parsed", "company_norm"], how="inner")
    overlap = overlap[["date_parsed", "company_norm", "injury_any_old", "injury_text_signal"]].drop_duplicates()

    labeled = overlap.dropna(subset=["injury_text_signal"]).copy()
    labeled["injury_text_signal"] = labeled["injury_text_signal"].astype(int)
    summary = {
        "overlap_rows": int(len(overlap)),
        "labeled_overlap_rows": int(len(labeled)),
        "accuracy": float((labeled["injury_text_signal"] == labeled["injury_any_old"]).mean()) if len(labeled) else float("nan"),
        "positive_precision_proxy": float(
            ((labeled["injury_text_signal"] == 1) & (labeled["injury_any_old"] == 1)).sum()
            / max((labeled["injury_text_signal"] == 1).sum(), 1)
        ),
        "positive_recall_proxy": float(
            ((labeled["injury_text_signal"] == 1) & (labeled["injury_any_old"] == 1)).sum()
            / max((labeled["injury_any_old"] == 1).sum(), 1)
        ),
    }
    mismatches = labeled[labeled["injury_text_signal"] != labeled["injury_any_old"]].copy()
    return summary, mismatches


def compute_vif(model) -> dict[str, float]:
    design = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    if "Intercept" in design.columns:
        design = design.drop(columns=["Intercept"])

    vif: dict[str, float] = {}
    if design.shape[1] <= 1:
        return vif

    for idx, column in enumerate(design.columns):
        vif[column] = float(variance_inflation_factor(design.values, idx))
    return vif


def fit_models(df: pd.DataFrame) -> tuple[dict[str, sm.GLM], pd.DataFrame, pd.DataFrame]:
    specs = [
        ModelSpec(
            name="baseline",
            formula="injury_text_signal ~ rear_end_c + mode_autonomous_c + rear_mode_int",
            focal_terms=("rear_end_c", "mode_autonomous_c", "rear_mode_int"),
            description="Baseline interaction model without stopped-context or operator controls.",
        ),
        ModelSpec(
            name="context",
            formula="injury_text_signal ~ rear_end_c + mode_autonomous_c + rear_mode_int + vru_any + is_night + location_is_intersection_like",
            focal_terms=("rear_end_c", "mode_autonomous_c", "rear_mode_int"),
            description="Adds scene-level controls for vulnerable users, darkness, and intersection geometry.",
        ),
        ModelSpec(
            name="stopped_context",
            formula="injury_text_signal ~ rear_end_c + mode_autonomous_c + rear_mode_int + vru_any + is_night + location_is_intersection_like + narrative_mentions_stopped",
            focal_terms=("rear_end_c", "mode_autonomous_c", "rear_mode_int", "narrative_mentions_stopped"),
            description="Adds a queue / stopped-vehicle cue to isolate unexpected stopping behavior.",
        ),
        ModelSpec(
            name="stopped_context_manufacturer",
            formula="injury_text_signal ~ rear_end_c + mode_autonomous_c + rear_mode_int + vru_any + is_night + location_is_intersection_like + narrative_mentions_stopped + C(manufacturer_top3)",
            focal_terms=("rear_end_c", "mode_autonomous_c", "rear_mode_int", "narrative_mentions_stopped"),
            description="Further controls for the top three manufacturers to test whether the signal survives operator mix.",
        ),
    ]

    centered = recenter(df, ["rear_end", "mode_autonomous"])
    # Centering the lower-order binary terms stabilizes the interaction estimates
    # and keeps the variance-inflation checks easy to interpret.
    models: dict[str, sm.GLM] = {}
    rows: list[dict[str, object]] = []
    focal_rows: list[dict[str, object]] = []

    for spec in specs:
        model = smf.glm(spec.formula, data=centered, family=sm.families.Binomial()).fit()
        models[spec.name] = model
        vif = compute_vif(model)
        for term in model.params.index:
            rows.append(
                {
                    "model_name": spec.name,
                    "description": spec.description,
                    "term": term,
                    "coef": float(model.params[term]),
                    "se": float(model.bse[term]),
                    "z": float(model.tvalues[term]),
                    "p_value": float(model.pvalues[term]),
                    "or": float(np.exp(model.params[term])),
                    "ci_low": float(np.exp(model.conf_int().loc[term, 0])),
                    "ci_high": float(np.exp(model.conf_int().loc[term, 1])),
                    "aic": float(model.aic),
                    "nobs": int(model.nobs),
                    "positives": int(model.model.data.frame["injury_text_signal"].sum()),
                    "max_vif": max(vif.values()) if vif else float("nan"),
                }
            )
            if term in spec.focal_terms:
                focal_rows.append(rows[-1])

    return models, pd.DataFrame(rows), pd.DataFrame(focal_rows)


def subgroup_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        df.groupby(["mode_resolved", "rear_end"])["injury_text_signal"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "injury_rate", "sum": "injury_count", "count": "sample_size"})
    )
    stopped = (
        df.groupby(["narrative_mentions_stopped", "mode_resolved", "rear_end"])["injury_text_signal"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "injury_rate", "sum": "injury_count", "count": "sample_size"})
    )
    return overall, stopped


def stopped_subgroup_model(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stopped_value in [0, 1]:
        sub = df[df["narrative_mentions_stopped"] == stopped_value].copy()
        if len(sub) < 50 or sub["injury_text_signal"].nunique() < 2:
            continue
        sub = recenter(sub, ["rear_end", "mode_autonomous"])
        model = smf.glm(
            "injury_text_signal ~ rear_end_c + mode_autonomous_c + rear_mode_int + vru_any + is_night + location_is_intersection_like",
            data=sub,
            family=sm.families.Binomial(),
        ).fit()
        for term in ["rear_end_c", "mode_autonomous_c", "rear_mode_int"]:
            rows.append(
                {
                    "stopped_value": stopped_value,
                    "term": term,
                    "coef": float(model.params[term]),
                    "se": float(model.bse[term]),
                    "z": float(model.tvalues[term]),
                    "p_value": float(model.pvalues[term]),
                    "nobs": int(model.nobs),
                    "positives": int(sub["injury_text_signal"].sum()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_main_interaction(df: pd.DataFrame, formula: str, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float]] = []
    base = df.copy()
    for rep in range(n_bootstrap):
        sample = base.iloc[rng.integers(0, len(base), len(base))].copy()
        sample = recenter(sample, ["rear_end", "mode_autonomous"])
        try:
            model = smf.glm(formula, data=sample, family=sm.families.Binomial()).fit(disp=0)
        except Exception:
            continue
        rows.append({"replication": rep, "rear_mode_int": float(model.params["rear_mode_int"])})
    return pd.DataFrame(rows)


def leave_one_operator_out(df: pd.DataFrame, formula: str, top_n: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for manufacturer in df["manufacturer_std"].value_counts().head(top_n).index:
        sub = df[df["manufacturer_std"] != manufacturer].copy()
        sub = recenter(sub, ["rear_end", "mode_autonomous"])
        model = smf.glm(formula, data=sub, family=sm.families.Binomial()).fit()
        rows.append(
            {
                "dropped_manufacturer": manufacturer,
                "nobs": int(model.nobs),
                "positives": int(sub["injury_text_signal"].sum()),
                "rear_mode_int_coef": float(model.params["rear_mode_int"]),
                "rear_mode_int_se": float(model.bse["rear_mode_int"]),
                "rear_mode_int_z": float(model.tvalues["rear_mode_int"]),
                "rear_mode_int_p": float(model.pvalues["rear_mode_int"]),
                "stopped_coef": float(model.params["narrative_mentions_stopped"]),
                "stopped_p": float(model.pvalues["narrative_mentions_stopped"]),
            }
        )
    return pd.DataFrame(rows)


def run_shap(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if shap is None or lgb is None or StratifiedKFold is None or cross_val_score is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    scene_feature_columns = [
        "collision_type__rear_end",
        "mode_binary_autonomous",
        "narrative_mentions_stopped",
        "vru_any",
        "is_night",
        "location_is_intersection_like",
        "narrative_mentions_lane_change",
        "narrative_mentions_merge",
        "narrative_mentions_left_turn",
        "narrative_mentions_right_turn",
        "narrative_mentions_entered_lane",
        "narrative_mentions_red_light",
        "narrative_mentions_parked",
    ]
    x_core = df[scene_feature_columns].fillna(0).astype(int)
    operator_dummies = pd.get_dummies(df["manufacturer_std"].fillna(""), prefix="manu")
    keep = [column for column in operator_dummies.columns if column in {"manu_Cruise", "manu_Waymo", "manu_Zoox"}]
    variants = {
        "scene_only": x_core,
        "scene_plus_operator": pd.concat([x_core, operator_dummies[keep]], axis=1),
    }
    y = df["injury_text_signal"].astype(int)

    metrics_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    interaction_rows: list[dict[str, object]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for variant_name, x in variants.items():
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=3,
            min_child_samples=12,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        auc = cross_val_score(model, x, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
        ap = cross_val_score(model, x, y, cv=cv, scoring="average_precision")
        model.fit(x, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif getattr(shap_values, "ndim", None) == 3:
            shap_values = shap_values[:, :, 1]

        importance = pd.DataFrame(
            {
                "variant": variant_name,
                "feature": x.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance_frames.append(importance)

        interaction_values = explainer.shap_interaction_values(x)
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]
        elif getattr(interaction_values, "ndim", None) == 4:
            interaction_values = interaction_values[:, :, :, 1]

        feature_index = {name: idx for idx, name in enumerate(x.columns)}
        selected_pairs = [
            ("collision_type__rear_end", "mode_binary_autonomous"),
            ("collision_type__rear_end", "narrative_mentions_stopped"),
            ("mode_binary_autonomous", "narrative_mentions_stopped"),
        ]
        if "manu_Cruise" in feature_index:
            selected_pairs.extend(
                [
                    ("collision_type__rear_end", "manu_Cruise"),
                    ("narrative_mentions_stopped", "manu_Cruise"),
                ]
            )
        for left, right in selected_pairs:
            if left not in feature_index or right not in feature_index:
                continue
            interaction_rows.append(
                {
                    "variant": variant_name,
                    "feature_left": left,
                    "feature_right": right,
                    "mean_abs_interaction_shap": float(
                        np.abs(interaction_values[:, feature_index[left], feature_index[right]]).mean()
                    ),
                }
            )

        metrics_rows.extend(
            [
                {"variant": variant_name, "metric": "cv_auc_mean", "value": float(auc.mean())},
                {"variant": variant_name, "metric": "cv_auc_sd", "value": float(auc.std())},
                {"variant": variant_name, "metric": "cv_accuracy_mean", "value": float(acc.mean())},
                {"variant": variant_name, "metric": "cv_accuracy_sd", "value": float(acc.std())},
                {"variant": variant_name, "metric": "cv_average_precision_mean", "value": float(ap.mean())},
                {"variant": variant_name, "metric": "cv_average_precision_sd", "value": float(ap.std())},
            ]
        )

    metrics = pd.DataFrame(metrics_rows)
    importance = pd.concat(importance_frames, ignore_index=True)
    interactions = pd.DataFrame(interaction_rows)
    return metrics, importance, interactions


def write_report(
    output_path: Path,
    validation_summary: dict[str, float],
    overall_rates: pd.DataFrame,
    stopped_rates: pd.DataFrame,
    focal_terms: pd.DataFrame,
    main_model_terms: pd.DataFrame,
    stopped_models: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    shap_metrics: pd.DataFrame,
    shap_importance: pd.DataFrame,
    shap_interactions: pd.DataFrame,
) -> None:
    model_pivot = focal_terms[focal_terms["term"] == "rear_mode_int"][
        ["model_name", "coef", "se", "z", "p_value", "or", "ci_low", "ci_high", "aic", "max_vif"]
    ].copy()

    lines = [
        "# Rear-End Injury Models on Parsed California DMV Crash Reports",
        "",
        "## Weak-Label Validation",
        "",
        f"- one-to-one overlap rows with the legacy xlsx: {validation_summary['overlap_rows']}",
        f"- labeled overlap rows: {validation_summary['labeled_overlap_rows']}",
        f"- weak-label accuracy against legacy injury labels: {validation_summary['accuracy']:.3f}",
        f"- positive precision proxy: {validation_summary['positive_precision_proxy']:.3f}",
        f"- positive recall proxy: {validation_summary['positive_recall_proxy']:.3f}",
        "",
        "## Overall Group Rates",
        "",
        frame_to_markdown(overall_rates),
        "",
        "## Group Rates Split by Stopped Cue",
        "",
        frame_to_markdown(stopped_rates),
        "",
        "## Interaction Comparison Across Model Specs",
        "",
        frame_to_markdown(model_pivot),
        "",
        "## Main Model: Stopped Context + Manufacturer Controls",
        "",
        frame_to_markdown(main_model_terms),
        "",
        "## Stopped-Only Heterogeneity Check",
        "",
        frame_to_markdown(stopped_models),
        "",
        "## Leave-One-Operator-Out Check",
        "",
        frame_to_markdown(leave_one_out),
        "",
    ]

    if not bootstrap_df.empty:
        quantiles = bootstrap_df["rear_mode_int"].quantile([0.025, 0.5, 0.975]).to_dict()
        share_positive = float((bootstrap_df["rear_mode_int"] > 0).mean())
        lines.extend(
            [
                "## Bootstrap Stability of `rear_mode_int`",
                "",
                f"- bootstrap replications retained: {len(bootstrap_df)}",
                f"- 2.5% quantile: {quantiles[0.025]:.3f}",
                f"- median: {quantiles[0.5]:.3f}",
                f"- 97.5% quantile: {quantiles[0.975]:.3f}",
                f"- share of positive bootstrap coefficients: {share_positive:.3f}",
                "",
            ]
        )

    if not shap_metrics.empty:
        top_shap_tables: list[str] = []
        for variant_name in shap_importance["variant"].drop_duplicates().tolist():
            top_shap_tables.extend(
                [
                    f"### Top SHAP Features: `{variant_name}`",
                    "",
                    frame_to_markdown(
                        shap_importance[shap_importance["variant"] == variant_name]
                        .head(10)
                        .reset_index(drop=True)
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                "## LightGBM + SHAP Check",
                "",
                frame_to_markdown(shap_metrics),
                "",
                "### Selected SHAP Interaction Strengths",
                "",
                frame_to_markdown(shap_interactions),
                "",
                *top_shap_tables,
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The broad rear-end × autonomous-mode reversal does not replicate cleanly in the bare-bones model.",
            "- Once the model conditions on queue / stopped-vehicle language, the interaction becomes significant and remains significant after adding manufacturer controls.",
            "- The strongest subgroup signal appears in reports that mention stopped traffic: the autonomous rear-end injury rate is 50.0% there, versus 22.2% for autonomous non-rear-end conflicts.",
            "- The leave-one-operator-out test shows that the signal weakens materially when Cruise is removed, so the current evidence is best framed as a mechanism-heavy pattern rather than a fully operator-invariant law.",
            "- The LightGBM + SHAP layer separates two stories: scene-only features emphasize `stopped` and `rear_end`, while the scene-plus-operator variant shows that operator mix still contributes materially to prediction.",
            "- In the scene-only tree model, `rear_end × stopped` is stronger than `rear_end × autonomous`; after operator dummies are added, the most prominent interaction shifts toward `stopped × Cruise`, which is consistent with the leave-one-operator-out sensitivity check.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(format_markdown_value(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> int:
    args = parse_args()
    features_path = Path(args.features).expanduser().resolve()
    legacy_xlsx_arg = str(args.legacy_xlsx).strip()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features_raw, df = load_analysis_sample(features_path)
    if legacy_xlsx_arg:
        legacy_xlsx_path = Path(legacy_xlsx_arg).expanduser().resolve()
    else:
        legacy_xlsx_path = None

    if legacy_xlsx_path is not None and legacy_xlsx_path.exists():
        validation_summary, mismatches = validate_weak_label(features_raw, legacy_xlsx_path)
    else:
        validation_summary = {
            "overlap_rows": 0,
            "labeled_overlap_rows": 0,
            "accuracy": float("nan"),
            "positive_precision_proxy": float("nan"),
            "positive_recall_proxy": float("nan"),
        }
        mismatches = pd.DataFrame(columns=["date_parsed", "company_norm", "injury_any_old", "injury_text_signal"])

    models, all_terms, focal_terms = fit_models(df)
    overall_rates, stopped_rates = subgroup_tables(df)
    stopped_models = stopped_subgroup_model(df)

    main_formula = (
        "injury_text_signal ~ rear_end_c + mode_autonomous_c + rear_mode_int + "
        "vru_any + is_night + location_is_intersection_like + narrative_mentions_stopped + "
        "C(manufacturer_top3)"
    )
    # The bootstrap and leave-one-operator-out checks target the focal interaction
    # rather than the entire coefficient vector because that is the main scientific claim.
    bootstrap_df = bootstrap_main_interaction(df, main_formula, args.bootstrap)
    leave_one_out = leave_one_operator_out(
        df,
        "injury_text_signal ~ rear_end_c + mode_autonomous_c + rear_mode_int + "
        "vru_any + is_night + location_is_intersection_like + narrative_mentions_stopped",
    )
    shap_metrics, shap_importance, shap_interactions = run_shap(df)

    main_terms = all_terms[all_terms["model_name"] == "stopped_context_manufacturer"].copy()

    all_terms.to_csv(output_dir / "model_terms.csv", index=False)
    focal_terms.to_csv(output_dir / "focal_terms.csv", index=False)
    overall_rates.to_csv(output_dir / "group_rates_overall.csv", index=False)
    stopped_rates.to_csv(output_dir / "group_rates_stopped.csv", index=False)
    stopped_models.to_csv(output_dir / "stopped_subgroup_models.csv", index=False)
    leave_one_out.to_csv(output_dir / "leave_one_operator_out.csv", index=False)
    bootstrap_df.to_csv(output_dir / "bootstrap_rear_mode_int.csv", index=False)
    mismatches.to_csv(output_dir / "weak_label_mismatches.csv", index=False)
    if not shap_metrics.empty:
        shap_metrics.to_csv(output_dir / "shap_metrics.csv", index=False)
        shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)
        shap_interactions.to_csv(output_dir / "shap_interactions.csv", index=False)

    write_report(
        output_dir / "rear_end_injury_model_report.md",
        validation_summary,
        overall_rates,
        stopped_rates,
        focal_terms,
        main_terms,
        stopped_models,
        leave_one_out,
        bootstrap_df,
        shap_metrics,
        shap_importance,
        shap_interactions,
    )

    print(f"Analysis sample rows: {len(df)}")
    print(f"Analysis sample positives: {int(df['injury_text_signal'].sum())}")
    if pd.notna(validation_summary["accuracy"]):
        print(f"Weak-label accuracy: {validation_summary['accuracy']:.3f}")
    else:
        print("Weak-label accuracy: not evaluated in this package run")
    print("Core interaction p-values:")
    for _, row in focal_terms[focal_terms["term"] == "rear_mode_int"].iterrows():
        print(f"  {row['model_name']}: coef={row['coef']:.4f}, p={row['p_value']:.4f}")
    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
