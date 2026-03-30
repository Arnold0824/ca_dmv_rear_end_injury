#!/usr/bin/env python3
"""Run the tree-based explanation layer for the rear-end injury study.

This script complements the regression analysis by:
1. engineering additional narrative mechanism features,
2. fitting multiple LightGBM variants,
3. exporting SHAP importance and interaction summaries, and
4. exporting dependence and local-explanation tables for interpretation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    import shap
    from sklearn.model_selection import StratifiedKFold, cross_val_score
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM/SHAP dependencies are unavailable: {exc}")


TOOLS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TOOLS_DIR.parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fit_rear_end_injury_models import DEFAULT_FEATURES, load_analysis_sample


DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results" / "explainable_ml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(DEFAULT_FEATURES), help="Prepared feature CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for SHAP outputs.")
    return parser.parse_args()


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    text = enriched["narrative_text"].fillna("").str.lower()

    # These text cues are intentionally transparent and easy to audit. They are
    # meant to approximate stopped-traffic and braking mechanisms, not to serve
    # as a full natural-language understanding pipeline.
    regex_map = {
        "mech_yielding": r"\byield(?:ed|ing)?\b",
        "mech_signal_wait": r"red light|green light|traffic light|stop sign|signal",
        "mech_stop_in_lane": (
            r"stopped in (?:the )?(?:lane|roadway|traffic lane|travel lane)|"
            r"was stopped in|had been stopped|came to a stop"
        ),
        "mech_queue_wait": r"\bqueue(?:d)?\b|waiting at|waiting for|stopped for approximately|line of cars",
        "mech_braking": r"\bbrak(?:e|ed|ing)\b|decelerat(?:e|ed|ing)|slowed? down",
        "mech_cut_in": r"cut-?in|unsafe maneuver|entered .* lane|crossing over a solid white line",
        "mech_rear_strike_phrase": r"rear-?ended|struck in the rear|struck from behind|hit from behind",
        "mech_pickup_dropoff": r"pick(?:ing)? up|drop-?off|passenger loading|curbside",
    }

    for column, pattern in regex_map.items():
        enriched[column] = text.str.contains(pattern, regex=True, na=False).astype(int)

    enriched["mech_stop_context_expanded"] = (
        enriched[
            [
                "narrative_mentions_stopped",
                "mech_yielding",
                "mech_signal_wait",
                "mech_stop_in_lane",
                "mech_queue_wait",
            ]
        ]
        .max(axis=1)
        .astype(int)
    )
    enriched["mech_brake_or_stop_transition"] = (
        enriched[["mech_braking", "mech_stop_in_lane", "mech_queue_wait"]].max(axis=1).astype(int)
    )
    enriched["mech_rear_stop_context"] = (
        enriched["rear_end"].astype(int) * enriched["mech_stop_context_expanded"].astype(int)
    )
    return enriched


def build_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scene_features = [
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
    mechanism_features = [
        "mech_yielding",
        "mech_signal_wait",
        "mech_stop_in_lane",
        "mech_queue_wait",
        "mech_braking",
        "mech_cut_in",
        "mech_rear_strike_phrase",
        "mech_pickup_dropoff",
        "mech_stop_context_expanded",
        "mech_brake_or_stop_transition",
    ]

    x_scene = df[scene_features].fillna(0).astype(int)
    x_mech = df[mechanism_features].fillna(0).astype(int)
    operator_dummies = pd.get_dummies(df["manufacturer_std"].fillna(""), prefix="manu")
    keep_ops = [col for col in operator_dummies.columns if col in {"manu_Cruise", "manu_Waymo", "manu_Zoox"}]
    x_operator = operator_dummies[keep_ops].astype(int) if keep_ops else pd.DataFrame(index=df.index)

    # The three variants isolate how much predictive signal comes from scene
    # structure alone, from additional mechanism text, and from operator mix.
    return {
        "scene_only": x_scene,
        "scene_plus_mechanism": pd.concat([x_scene, x_mech], axis=1),
        "scene_plus_mechanism_operator": pd.concat([x_scene, x_mech, x_operator], axis=1),
    }


def build_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=9,
        max_depth=4,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.6,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )


def normalize_shap_values(raw_values: object) -> np.ndarray:
    values = raw_values
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", None) == 3:
        values = values[:, :, 1]
    return np.asarray(values)


def normalize_interaction_values(raw_values: object) -> np.ndarray:
    values = raw_values
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", None) == 4:
        values = values[:, :, :, 1]
    return np.asarray(values)


def normalize_expected_value(expected_value: object) -> float:
    if isinstance(expected_value, list):
        return float(expected_value[1])
    arr = np.asarray(expected_value)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 2:
        return float(arr[1])
    return float(arr.reshape(-1)[0])


def feature_catalog(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        series = pd.to_numeric(df[feature], errors="coerce").fillna(0).astype(int)
        subset = df[series == 1]
        rows.append(
            {
                "feature": feature,
                "count": int(series.sum()),
                "prevalence": float(series.mean()),
                "injury_rate_when_present": float(subset["injury_text_signal"].mean()) if len(subset) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "feature"], ascending=[False, True]).reset_index(drop=True)


def dependence_summary(
    variant_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    pred: np.ndarray,
    shap_values: np.ndarray,
    selected_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in selected_features:
        if feature not in x.columns:
            continue
        feature_idx = x.columns.get_loc(feature)
        frame = pd.DataFrame(
            {
                "feature_value": pd.to_numeric(x[feature], errors="coerce").fillna(0).astype(int),
                "pred_prob": pred,
                "shap_value": shap_values[:, feature_idx],
                "injury": y.to_numpy(),
            }
        )
        for feature_value, sub in frame.groupby("feature_value"):
            rows.append(
                {
                    "variant": variant_name,
                    "feature": feature,
                    "feature_value": int(feature_value),
                    "count": int(len(sub)),
                    "injury_rate": float(sub["injury"].mean()),
                    "mean_pred_prob": float(sub["pred_prob"].mean()),
                    "mean_shap": float(sub["shap_value"].mean()),
                    "median_shap": float(sub["shap_value"].median()),
                    "mean_abs_shap": float(sub["shap_value"].abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def interaction_exports(
    variant_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    pred: np.ndarray,
    interaction_values: np.ndarray,
    selected_pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_index = {name: idx for idx, name in enumerate(x.columns)}
    summary_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []

    for left, right in selected_pairs:
        if left not in feature_index or right not in feature_index:
            continue
        left_idx = feature_index[left]
        right_idx = feature_index[right]
        series = interaction_values[:, left_idx, right_idx]
        summary_rows.append(
            {
                "variant": variant_name,
                "feature_left": left,
                "feature_right": right,
                "mean_interaction_shap": float(series.mean()),
                "mean_abs_interaction_shap": float(np.abs(series).mean()),
            }
        )

        frame = pd.DataFrame(
            {
                "left_value": pd.to_numeric(x[left], errors="coerce").fillna(0).astype(int),
                "right_value": pd.to_numeric(x[right], errors="coerce").fillna(0).astype(int),
                "interaction_shap": series,
                "pred_prob": pred,
                "injury": y.to_numpy(),
            }
        )
        for (left_value, right_value), sub in frame.groupby(["left_value", "right_value"]):
            cell_rows.append(
                {
                    "variant": variant_name,
                    "feature_left": left,
                    "feature_right": right,
                    "left_value": int(left_value),
                    "right_value": int(right_value),
                    "count": int(len(sub)),
                    "injury_rate": float(sub["injury"].mean()),
                    "mean_pred_prob": float(sub["pred_prob"].mean()),
                    "mean_interaction_shap": float(sub["interaction_shap"].mean()),
                    "mean_abs_interaction_shap": float(sub["interaction_shap"].abs().mean()),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(cell_rows)


def select_local_cases(
    df: pd.DataFrame,
    pred: np.ndarray,
) -> pd.DataFrame:
    frame = df.copy()
    frame["pred_prob"] = pred
    text = frame["narrative_text"].fillna("").str.lower()
    frame["clean_negative_text"] = text.str.contains(
        (
            r"no injuries|no injury|no reported injuries|no reported injury|"
            r"neither party reported injuries|there were no reported injuries|"
            r"did not report any injuries"
        ),
        regex=True,
        na=False,
    ).astype(int)
    frame["clean_positive_text"] = (
        text.str.contains(
        (
            r"reported injuries|reported injury|reported minor headache|reported back pain|"
            r"headache and neck soreness|sought medical attention|declined medical treatment|"
            r"complained of pain|operator reported injuries"
        ),
        regex=True,
        na=False,
    ) & (frame["clean_negative_text"] == 0)).astype(int)

    case_specs = [
        (
            "av_rear_stopped_injury",
            lambda d: (
                (d["mode_resolved"] == "autonomous")
                & (d["rear_end"] == 1)
                & (d["mech_stop_context_expanded"] == 1)
                & (d["injury_text_signal"] == 1)
            ),
        ),
        (
            "av_rear_stopped_no_injury",
            lambda d: (
                (d["mode_resolved"] == "autonomous")
                & (d["rear_end"] == 1)
                & (d["mech_stop_context_expanded"] == 1)
                & (d["injury_text_signal"] == 0)
            ),
        ),
        (
            "conv_rear_stopped_injury",
            lambda d: (
                (d["mode_resolved"] == "conventional")
                & (d["rear_end"] == 1)
                & (d["mech_stop_context_expanded"] == 1)
                & (d["injury_text_signal"] == 1)
            ),
        ),
        (
            "av_rear_nonstopped_no_injury",
            lambda d: (
                (d["mode_resolved"] == "autonomous")
                & (d["rear_end"] == 1)
                & (d["mech_stop_context_expanded"] == 0)
                & (d["injury_text_signal"] == 0)
            ),
        ),
        (
            "av_nonrear_stopped_injury",
            lambda d: (
                (d["mode_resolved"] == "autonomous")
                & (d["rear_end"] == 0)
                & (d["mech_stop_context_expanded"] == 1)
                & (d["injury_text_signal"] == 1)
            ),
        ),
    ]

    picks: list[pd.Series] = []
    for case_group, mask_fn in case_specs:
        subset = frame[mask_fn(frame)].copy()
        if subset.empty:
            continue
        if case_group.endswith("no_injury"):
            consistent = subset[subset["clean_negative_text"] == 1].copy()
            if consistent.empty:
                continue
            subset = consistent
        elif case_group.endswith("injury"):
            consistent = subset[(subset["clean_positive_text"] == 1) & (subset["clean_negative_text"] == 0)].copy()
            if consistent.empty:
                continue
            subset = consistent
        target = subset["pred_prob"].median()
        subset["distance_to_group_median"] = (subset["pred_prob"] - target).abs()
        row = subset.sort_values(["distance_to_group_median", "pred_prob"]).iloc[0].copy()
        row["case_group"] = case_group
        picks.append(row)

    if not picks:
        return pd.DataFrame()
    return pd.DataFrame(picks).reset_index(drop=True)


def local_explanations(
    cases: pd.DataFrame,
    x: pd.DataFrame,
    shap_values: np.ndarray,
    expected_value: float,
    top_k: int = 8,
) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    x_local = x.reset_index(drop=True)
    case_frame = cases.reset_index(drop=True)
    for _, case in case_frame.iterrows():
        row_position = int(case["row_position"])
        feature_values = x_local.loc[row_position]
        shap_row = shap_values[row_position]
        order = np.argsort(np.abs(shap_row))[::-1][:top_k]
        for rank, feature_idx in enumerate(order, start=1):
            feature = x_local.columns[feature_idx]
            rows.append(
                {
                    "case_group": case["case_group"],
                    "report_id": case["report_id"],
                    "report_title": case["report_title"],
                    "manufacturer_std": case["manufacturer_std"],
                    "mode_resolved": case["mode_resolved"],
                    "injury_text_signal": int(case["injury_text_signal"]),
                    "pred_prob": float(case["pred_prob"]),
                    "expected_value": expected_value,
                    "rank": rank,
                    "feature": feature,
                    "feature_value": float(feature_values.iloc[feature_idx]),
                    "shap_value": float(shap_row[feature_idx]),
                    "narrative_excerpt": str(case["narrative_text"])[:450],
                }
            )
    return pd.DataFrame(rows)


def write_readme(output_dir: Path, metrics: pd.DataFrame) -> None:
    lines = [
        "# Tree-Based Explanation Outputs",
        "",
        "This directory contains the standalone LightGBM + SHAP explanation outputs for the rear-end injury analysis.",
        "",
        "## Main files",
        "",
        "- `variant_metrics.csv`: cross-validated performance by variant.",
        "- `feature_catalog.csv`: prevalence and injury rate for engineered mechanism features.",
        "- `shap_importance.csv`: global mean absolute SHAP values.",
        "- `shap_interactions.csv`: selected pairwise interaction SHAP summaries.",
        "- `interaction_cell_summary.csv`: 2x2 cell summaries for selected interaction pairs.",
        "- `dependence_summary.csv`: binary dependence summaries for selected features.",
        "- `local_case_selection.csv`: representative cases used for local explanation.",
        "- `local_case_explanations.csv`: top feature contributions for the selected local cases.",
        "",
        "## Variants",
        "",
    ]
    for _, row in metrics.pivot(index="variant", columns="metric", values="value").reset_index().iterrows():
        lines.append(
            f"- `{row['variant']}`: CV AUC={row['cv_auc_mean']:.3f}, "
            f"accuracy={row['cv_accuracy_mean']:.3f}, average precision={row['cv_average_precision_mean']:.3f}"
        )
    lines.append("")
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, df = load_analysis_sample(Path(args.features).expanduser().resolve())
    df = add_mechanism_features(df).reset_index(drop=False).rename(columns={"index": "original_index"})

    variants = build_variants(df)
    y = df["injury_text_signal"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    interaction_frames: list[pd.DataFrame] = []
    interaction_cell_frames: list[pd.DataFrame] = []
    dependence_frames: list[pd.DataFrame] = []
    local_case_selection = pd.DataFrame()
    local_case_frame = pd.DataFrame()

    feature_catalog_df = feature_catalog(
        df,
        [
            "mech_yielding",
            "mech_signal_wait",
            "mech_stop_in_lane",
            "mech_queue_wait",
            "mech_braking",
            "mech_cut_in",
            "mech_rear_strike_phrase",
            "mech_pickup_dropoff",
            "mech_stop_context_expanded",
            "mech_brake_or_stop_transition",
        ],
    )

    selected_dependence_features = [
        "collision_type__rear_end",
        "mode_binary_autonomous",
        "narrative_mentions_stopped",
        "mech_stop_context_expanded",
        "mech_signal_wait",
        "mech_yielding",
        "mech_braking",
        "mech_cut_in",
        "manu_Cruise",
    ]

    selected_pairs = [
        ("collision_type__rear_end", "mode_binary_autonomous"),
        ("collision_type__rear_end", "narrative_mentions_stopped"),
        ("collision_type__rear_end", "mech_stop_context_expanded"),
        ("collision_type__rear_end", "mech_signal_wait"),
        ("collision_type__rear_end", "mech_yielding"),
        ("mech_stop_context_expanded", "manu_Cruise"),
        ("collision_type__rear_end", "manu_Cruise"),
    ]

    for variant_name, x in variants.items():
        model = build_model()
        auc = cross_val_score(model, x, y, cv=cv, scoring="roc_auc")
        acc = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
        ap = cross_val_score(model, x, y, cv=cv, scoring="average_precision")
        model.fit(x, y)
        pred = model.predict_proba(x)[:, 1]

        explainer = shap.TreeExplainer(model)
        shap_values = normalize_shap_values(explainer.shap_values(x))
        interaction_values = normalize_interaction_values(explainer.shap_interaction_values(x))
        expected_value = normalize_expected_value(explainer.expected_value)

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

        importance = pd.DataFrame(
            {
                "variant": variant_name,
                "feature": x.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance_frames.append(importance)

        interaction_summary, interaction_cells = interaction_exports(
            variant_name, x, y, pred, interaction_values, selected_pairs
        )
        interaction_frames.append(interaction_summary)
        interaction_cell_frames.append(interaction_cells)

        dependence_frames.append(
            dependence_summary(variant_name, x, y, pred, shap_values, selected_dependence_features)
        )

        prediction_frame = df[
            [
                "original_index",
                "report_id",
                "report_title",
                "manufacturer_std",
                "mode_resolved",
                "injury_text_signal",
                "rear_end",
                "mech_stop_context_expanded",
                "narrative_mentions_injury",
                "narrative_mentions_no_injury",
                "narrative_text",
            ]
        ].copy()
        prediction_frame["row_position"] = np.arange(len(prediction_frame))
        prediction_frame["pred_prob"] = pred

        if variant_name == "scene_plus_mechanism_operator":
            local_case_selection = select_local_cases(prediction_frame, pred)
            if not local_case_selection.empty:
                local_case_frame = local_explanations(
                    local_case_selection,
                    x,
                    shap_values,
                    expected_value,
                )

    metrics = pd.DataFrame(metrics_rows)
    shap_importance = pd.concat(importance_frames, ignore_index=True)
    shap_interactions = pd.concat(interaction_frames, ignore_index=True)
    interaction_cells = pd.concat(interaction_cell_frames, ignore_index=True)
    dependence = pd.concat(dependence_frames, ignore_index=True)

    metrics.to_csv(output_dir / "variant_metrics.csv", index=False)
    feature_catalog_df.to_csv(output_dir / "feature_catalog.csv", index=False)
    shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)
    shap_interactions.to_csv(output_dir / "shap_interactions.csv", index=False)
    interaction_cells.to_csv(output_dir / "interaction_cell_summary.csv", index=False)
    dependence.to_csv(output_dir / "dependence_summary.csv", index=False)
    if not local_case_selection.empty:
        local_case_selection.to_csv(output_dir / "local_case_selection.csv", index=False)
    if not local_case_frame.empty:
        local_case_frame.to_csv(output_dir / "local_case_explanations.csv", index=False)
    write_readme(output_dir, metrics)

    print("Saved deep SHAP outputs to", output_dir)
    print(metrics.pivot(index="variant", columns="metric", values="value")[["cv_auc_mean", "cv_accuracy_mean"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
