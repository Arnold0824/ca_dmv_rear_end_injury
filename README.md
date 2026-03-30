# California DMV AV Crash Injury Reproducibility Package

This package contains the processed data tables, statistical outputs, figure-building scripts, and manuscript assets used for the study of rear-end injury patterns in California autonomous-vehicle crash reports.

## Package structure

- `data/public_source_manifest/`
  Public-record manifest for the California DMV collision-report PDFs used in the study.
- `data/processed/`
  Parsed and engineered CSV tables used by the inferential and explainable-modeling scripts.
- `results/inferential/`
  Logistic-model summaries, grouped rates, bootstrap results, and operator leave-one-out outputs.
- `results/explainable_ml/`
  LightGBM-SHAP metrics, feature summaries, interaction tables, local explanations, and supporting diagnostic figures.
- `scripts/`
  Reproducible code organized by data collection, data processing, modeling, and manuscript-support tasks.
- `manuscript_assets/`
  Figures and LaTeX table files used in the manuscript.
- `docs/`
  Short notes describing data sources, run order, and feature coverage.

## Important notes

- The original California DMV collision-report PDFs are not redistributed in this package. They remain publicly available from the California DMV.
- Local absolute file paths from the working environment were removed from the shared CSV files and replaced with sanitized relative references.
- The legacy structured California DMV crash table used for weak-label validation is not redistributed here. If you have access to that file, you can pass it to the inferential modeling script with `--legacy-xlsx`.

## Recommended run order

```bash
python scripts/data_collection/fetch_ca_dmv_collision_reports.py
python scripts/data_processing/parse_ca_dmv_collision_reports.py
python scripts/data_processing/engineer_collision_features.py
python scripts/modeling/fit_rear_end_injury_models.py
python scripts/modeling/run_tree_explanations.py
```

Figure builders for the manuscript are located under `scripts/manuscript_support/`.

## Raw data source

California Department of Motor Vehicles autonomous-vehicle collision reports:

https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-collision-reports/
