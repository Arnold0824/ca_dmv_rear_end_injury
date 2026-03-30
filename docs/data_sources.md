# Data Sources

## Original public source

The underlying crash reports come from the California Department of Motor Vehicles autonomous-vehicle collision-report archive:

https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-collision-reports/

The archive provides public PDF reports for the modern reporting window. Earlier archived reports must be requested directly from the DMV.

## Files included in this package

- `data/public_source_manifest/ca_dmv_collision_report_manifest.csv`
  Manifest of the public PDF reports used in the study.
- `data/processed/parsed_reports_raw_wide.csv`
  One row per report with the raw PDF widget fields.
- `data/processed/parsed_reports_normalized.csv`
  Harmonized field names for common crash-report variables.
- `data/processed/parsed_widgets_long.csv`
  Long-format widget extraction table for auditability.
- `data/processed/parsed_variable_dictionary.csv`
  Coverage and semantic summary of the parsed fields.
- `data/processed/engineered_features.csv`
  Model-ready feature table used in the inferential and explainable-modeling steps.
- `data/processed/engineered_feature_dictionary.csv`
  Feature definitions and availability summary.

## Not redistributed

The legacy structured California DMV crash table used for weak-label validation is not redistributed in this package.
