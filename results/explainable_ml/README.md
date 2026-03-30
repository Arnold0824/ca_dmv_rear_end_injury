# Tree-Based Explanation Outputs

This directory contains the standalone LightGBM + SHAP explanation outputs for the rear-end injury analysis.

## Main files

- `variant_metrics.csv`: cross-validated performance by variant.
- `feature_catalog.csv`: prevalence and injury rate for engineered mechanism features.
- `shap_importance.csv`: global mean absolute SHAP values.
- `shap_interactions.csv`: selected pairwise interaction SHAP summaries.
- `interaction_cell_summary.csv`: 2x2 cell summaries for selected interaction pairs.
- `dependence_summary.csv`: binary dependence summaries for selected features.
- `local_case_selection.csv`: representative cases used for local explanation.
- `local_case_explanations.csv`: top feature contributions for the selected local cases.

## Variants

- `scene_only`: CV AUC=0.606, accuracy=0.585, average precision=0.480
- `scene_plus_mechanism`: CV AUC=0.572, accuracy=0.579, average precision=0.428
- `scene_plus_mechanism_operator`: CV AUC=0.665, accuracy=0.615, average precision=0.477
