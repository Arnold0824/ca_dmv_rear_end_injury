# Workflow

## 1. Collect public reports

`scripts/data_collection/fetch_ca_dmv_collision_reports.py`

Downloads the publicly linked California DMV collision-report PDFs and writes a manifest.

## 2. Parse report forms

`scripts/data_processing/parse_ca_dmv_collision_reports.py`

Extracts raw widget-level fields from the PDFs and writes raw, normalized, and long-format audit tables.

## 3. Engineer analysis features

`scripts/data_processing/engineer_collision_features.py`

Builds the model-ready feature table used by the inferential and tree-based analyses.

## 4. Estimate inferential models

`scripts/modeling/fit_rear_end_injury_models.py`

Fits the nested logistic-regression models, grouped-rate summaries, bootstrap checks, and operator leave-one-out analysis.

## 5. Run tree-based explanations

`scripts/modeling/run_tree_explanations.py`

Fits the LightGBM variants and exports SHAP-based feature, interaction, and local-explanation outputs.

## 6. Build manuscript assets

`scripts/manuscript_support/`

Contains figure builders used to regenerate the manuscript figures and LaTeX tables from the processed data and analysis outputs.
