# Rear-End Injury Models on Parsed California DMV Crash Reports

## Weak-Label Validation

- one-to-one overlap rows with the legacy xlsx: 0
- labeled overlap rows: 0
- weak-label accuracy against legacy injury labels: nan
- positive precision proxy: nan
- positive recall proxy: nan

## Overall Group Rates

| mode_resolved | rear_end | injury_rate | injury_count | sample_size |
| --- | --- | --- | --- | --- |
| autonomous | 0 | 0.2824 | 24 | 85 |
| autonomous | 1 | 0.3761 | 44 | 117 |
| conventional | 0 | 0.3253 | 27 | 83 |
| conventional | 1 | 0.2545 | 14 | 55 |

## Group Rates Split by Stopped Cue

| narrative_mentions_stopped | mode_resolved | rear_end | injury_rate | injury_count | sample_size |
| --- | --- | --- | --- | --- | --- |
| 0 | autonomous | 0 | 0.3103 | 18 | 58 |
| 0 | autonomous | 1 | 0.3014 | 22 | 73 |
| 0 | conventional | 0 | 0.3030 | 20 | 66 |
| 0 | conventional | 1 | 0.1304 | 3 | 23 |
| 1 | autonomous | 0 | 0.2222 | 6 | 27 |
| 1 | autonomous | 1 | 0.5000 | 22 | 44 |
| 1 | conventional | 0 | 0.4118 | 7 | 17 |
| 1 | conventional | 1 | 0.3438 | 11 | 32 |

## Interaction Comparison Across Model Specs

| model_name | coef | se | z | p_value | or | ci_low | ci_high | aic | max_vif |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.7716 | 0.4952 | 1.5581 | 0.1192 | 2.1631 | 0.8196 | 5.7092 | 431.2235 | 1.0373 |
| context | 0.9264 | 0.5069 | 1.8277 | 0.0676 | 2.5255 | 0.9351 | 6.8203 | 426.0556 | 1.4502 |
| stopped_context | 1.1340 | 0.5220 | 2.1725 | 0.0298 | 3.1081 | 1.1173 | 8.6457 | 423.4337 | 1.6420 |
| stopped_context_manufacturer | 1.1698 | 0.5521 | 2.1187 | 0.0341 | 3.2214 | 1.0916 | 9.5066 | 394.7177 | 2.5123 |

## Main Model: Stopped Context + Manufacturer Controls

| model_name | description | term | coef | se | z | p_value | or | ci_low | ci_high | aic | nobs | positives | max_vif |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | Intercept | -0.3924 | 0.2570 | -1.5267 | 0.1268 | 0.6755 | 0.4082 | 1.1178 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | C(manufacturer_top3)[T.Other] | -2.5725 | 0.7667 | -3.3551 | 0.0008 | 0.0763 | 0.0170 | 0.3431 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | C(manufacturer_top3)[T.Waymo] | -0.8332 | 0.4376 | -1.9041 | 0.0569 | 0.4347 | 0.1844 | 1.0247 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | C(manufacturer_top3)[T.Zoox] | 0.5134 | 0.3314 | 1.5490 | 0.1214 | 1.6709 | 0.8727 | 3.1993 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | rear_end_c | 0.3433 | 0.2787 | 1.2319 | 0.2180 | 1.4096 | 0.8164 | 2.4340 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | mode_autonomous_c | 0.1609 | 0.2898 | 0.5553 | 0.5787 | 1.1746 | 0.6656 | 2.0729 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | rear_mode_int | 1.1698 | 0.5521 | 2.1187 | 0.0341 | 3.2214 | 1.0916 | 9.5066 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | vru_any | 0.4740 | 0.4262 | 1.1122 | 0.2661 | 1.6065 | 0.6967 | 3.7040 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | is_night | -0.1462 | 0.2601 | -0.5619 | 0.5742 | 0.8640 | 0.5189 | 1.4386 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | location_is_intersection_like | -0.9056 | 0.3016 | -3.0023 | 0.0027 | 0.4043 | 0.2239 | 0.7302 | 394.7177 | 340 | 109 | 2.5123 |
| stopped_context_manufacturer | Further controls for the top three manufacturers to test whether the signal survives operator mix. | narrative_mentions_stopped | 0.7097 | 0.2878 | 2.4661 | 0.0137 | 2.0334 | 1.1568 | 3.5744 | 394.7177 | 340 | 109 | 2.5123 |

## Stopped-Only Heterogeneity Check

| stopped_value | term | coef | se | z | p_value | nobs | positives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | rear_end_c | -0.2272 | 0.3699 | -0.6141 | 0.5391 | 220 | 63 |
| 0 | mode_autonomous_c | 0.5198 | 0.3734 | 1.3921 | 0.1639 | 220 | 63 |
| 0 | rear_mode_int | 1.0186 | 0.7872 | 1.2939 | 0.1957 | 220 | 63 |
| 1 | rear_end_c | 0.8008 | 0.4513 | 1.7742 | 0.0760 | 120 | 46 |
| 1 | mode_autonomous_c | 0.0632 | 0.4064 | 0.1555 | 0.8764 | 120 | 46 |
| 1 | rear_mode_int | 2.0752 | 0.9047 | 2.2938 | 0.0218 | 120 | 46 |

## Leave-One-Operator-Out Check

| dropped_manufacturer | nobs | positives | rear_mode_int_coef | rear_mode_int_se | rear_mode_int_z | rear_mode_int_p | stopped_coef | stopped_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cruise | 187 | 52 | 0.8766 | 0.6995 | 1.2532 | 0.2101 | 0.7020 | 0.0483 |
| Zoox | 252 | 70 | 1.7362 | 0.7202 | 2.4106 | 0.0159 | 0.6564 | 0.0430 |
| Waymo | 284 | 98 | 1.1416 | 0.5591 | 2.0418 | 0.0412 | 0.5616 | 0.0491 |
| Apple | 325 | 109 | 1.1091 | 0.5267 | 2.1060 | 0.0352 | 0.5882 | 0.0224 |
| Pony.ai | 328 | 109 | 1.1772 | 0.5286 | 2.2270 | 0.0259 | 0.5652 | 0.0292 |
| Lyft, Inc. | 335 | 109 | 1.0262 | 0.5246 | 1.9560 | 0.0505 | 0.5367 | 0.0363 |

## Bootstrap Stability of `rear_mode_int`

- bootstrap replications retained: 300
- 2.5% quantile: 0.095
- median: 1.222
- 97.5% quantile: 2.415
- share of positive bootstrap coefficients: 0.983

## LightGBM + SHAP Check

| variant | metric | value |
| --- | --- | --- |
| scene_only | cv_auc_mean | 0.6014 |
| scene_only | cv_auc_sd | 0.0895 |
| scene_only | cv_accuracy_mean | 0.5676 |
| scene_only | cv_accuracy_sd | 0.0523 |
| scene_only | cv_average_precision_mean | 0.4782 |
| scene_only | cv_average_precision_sd | 0.0942 |
| scene_plus_operator | cv_auc_mean | 0.6731 |
| scene_plus_operator | cv_auc_sd | 0.0678 |
| scene_plus_operator | cv_accuracy_mean | 0.6059 |
| scene_plus_operator | cv_accuracy_sd | 0.0496 |
| scene_plus_operator | cv_average_precision_mean | 0.4734 |
| scene_plus_operator | cv_average_precision_sd | 0.0914 |

### Selected SHAP Interaction Strengths

| variant | feature_left | feature_right | mean_abs_interaction_shap |
| --- | --- | --- | --- |
| scene_only | collision_type__rear_end | mode_binary_autonomous | 0.0263 |
| scene_only | collision_type__rear_end | narrative_mentions_stopped | 0.0538 |
| scene_only | mode_binary_autonomous | narrative_mentions_stopped | 0.0044 |
| scene_plus_operator | collision_type__rear_end | mode_binary_autonomous | 0.0972 |
| scene_plus_operator | collision_type__rear_end | narrative_mentions_stopped | 0.0382 |
| scene_plus_operator | mode_binary_autonomous | narrative_mentions_stopped | 0.0085 |
| scene_plus_operator | collision_type__rear_end | manu_Cruise | 0.0314 |
| scene_plus_operator | narrative_mentions_stopped | manu_Cruise | 0.1150 |

### Top SHAP Features: `scene_only`

| variant | feature | mean_abs_shap |
| --- | --- | --- |
| scene_only | location_is_intersection_like | 0.3435 |
| scene_only | narrative_mentions_stopped | 0.2278 |
| scene_only | narrative_mentions_entered_lane | 0.1396 |
| scene_only | collision_type__rear_end | 0.1345 |
| scene_only | vru_any | 0.1145 |
| scene_only | narrative_mentions_red_light | 0.1085 |
| scene_only | narrative_mentions_parked | 0.0859 |
| scene_only | is_night | 0.0767 |
| scene_only | mode_binary_autonomous | 0.0596 |
| scene_only | narrative_mentions_left_turn | 0.0439 |

### Top SHAP Features: `scene_plus_operator`

| variant | feature | mean_abs_shap |
| --- | --- | --- |
| scene_plus_operator | manu_Cruise | 0.7085 |
| scene_plus_operator | manu_Zoox | 0.6573 |
| scene_plus_operator | narrative_mentions_stopped | 0.4434 |
| scene_plus_operator | location_is_intersection_like | 0.3511 |
| scene_plus_operator | collision_type__rear_end | 0.2771 |
| scene_plus_operator | vru_any | 0.1119 |
| scene_plus_operator | is_night | 0.1051 |
| scene_plus_operator | mode_binary_autonomous | 0.0920 |
| scene_plus_operator | narrative_mentions_entered_lane | 0.0884 |
| scene_plus_operator | narrative_mentions_right_turn | 0.0846 |

## Interpretation

- The broad rear-end × autonomous-mode reversal does not replicate cleanly in the bare-bones model.
- Once the model conditions on queue / stopped-vehicle language, the interaction becomes significant and remains significant after adding manufacturer controls.
- The strongest subgroup signal appears in reports that mention stopped traffic: the autonomous rear-end injury rate is 50.0% there, versus 22.2% for autonomous non-rear-end conflicts.
- The leave-one-operator-out test shows that the signal weakens materially when Cruise is removed, so the current evidence is best framed as a mechanism-heavy pattern rather than a fully operator-invariant law.
- The LightGBM + SHAP layer separates two stories: scene-only features emphasize `stopped` and `rear_end`, while the scene-plus-operator variant shows that operator mix still contributes materially to prediction.
- In the scene-only tree model, `rear_end × stopped` is stronger than `rear_end × autonomous`; after operator dummies are added, the most prominent interaction shifts toward `stopped × Cruise`, which is consistent with the leave-one-operator-out sensitivity check.
