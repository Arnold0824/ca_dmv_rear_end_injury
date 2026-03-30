#!/usr/bin/env python3
"""Build the engineered feature table used by the inferential and tree models."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PACKAGE_ROOT / "data" / "processed" / "parsed_reports_normalized.csv"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "processed" / "engineered_features.csv"
DEFAULT_DICTIONARY = PACKAGE_ROOT / "data" / "processed" / "engineered_feature_dictionary.csv"
DEFAULT_PROFILE = PACKAGE_ROOT / "docs" / "feature_profile.md"

MULTI_VALUE_COLUMNS = {
    "collision_type_resolved_labels": "collision_type",
    "weather_vehicle1_labels": "weather_v1",
    "weather_vehicle2_labels": "weather_v2",
    "lighting_vehicle1_labels": "lighting_v1",
    "lighting_vehicle2_labels": "lighting_v2",
    "roadway_surface_vehicle1_labels": "roadway_surface_v1",
    "roadway_surface_vehicle2_labels": "roadway_surface_v2",
    "movement_vehicle1_labels": "movement_v1",
    "movement_vehicle2_labels": "movement_v2",
    "road_conditions_vehicle1_labels": "road_conditions_v1",
    "road_conditions_vehicle2_labels": "road_conditions_v2",
    "other_associated_factor_labels": "other_factor",
    "vehicle1_damage_zones": "damage_zone_v1",
}
DAMAGE_ORDINAL = {
    "none": 0,
    "minor": 1,
    "moderate": 2,
    "major": 3,
    "unknown": -1,
}
NARRATIVE_PATTERNS = {
    "narrative_mentions_injury": re.compile(
        r"reported injuries|complained of bodily injury|sought medical attention|transported to (?:a )?hospital|"
        r"treated at (?:the )?scene|minor headache|neck stiffness|back pain|neck pain|reported injuries|"
        r"reported injury|sustained injuries|was injured|were injured|fatal|deceased",
        re.I,
    ),
    "narrative_mentions_no_injury": re.compile(
        r"no injuries? were reported|there were no injuries|no one was injured|"
        r"neither party reported injuries?|no injuries? at the scene|no injuries? and police|"
        r"declined medical treatment",
        re.I,
    ),
    "narrative_mentions_police": re.compile(r"\bpolice\b|law enforcement", re.I),
    "narrative_mentions_no_police": re.compile(r"\bpolice were not called\b|\bno police report\b", re.I),
    "narrative_mentions_ems": re.compile(r"emergency services|ambulance|paramedic|medical treatment|hospital", re.I),
    "narrative_mentions_pedestrian": re.compile(r"\bpedestrian\b", re.I),
    "narrative_mentions_bicyclist": re.compile(r"\bbicycl(?:e|ist)\b", re.I),
    "narrative_mentions_scooter": re.compile(r"\bscooter(?:ist)?\b", re.I),
    "narrative_mentions_motorcycle": re.compile(r"\bmotorcycle|motorcyclist\b", re.I),
    "narrative_mentions_rear": re.compile(r"rear[- ]end|struck .*rear|hit .*rear|rear bumper", re.I),
    "narrative_mentions_left_turn": re.compile(r"left turn|turning left|making a left", re.I),
    "narrative_mentions_right_turn": re.compile(r"right turn|turning right|making a right", re.I),
    "narrative_mentions_u_turn": re.compile(r"u-turn|u turn", re.I),
    "narrative_mentions_lane_change": re.compile(r"changing lanes|lane change", re.I),
    "narrative_mentions_merge": re.compile(r"\bmerge|merging\b", re.I),
    "narrative_mentions_backing": re.compile(r"\bbacking\b|\breversed\b|\breverse\b", re.I),
    "narrative_mentions_parked": re.compile(r"\bparked\b|parking garage|parking lot|pulling out of parking", re.I),
    "narrative_mentions_stopped": re.compile(r"\bstopped\b|stop(?:ped)? at", re.I),
    "narrative_mentions_red_light": re.compile(r"red light", re.I),
    "narrative_mentions_green_light": re.compile(r"green light", re.I),
    "narrative_mentions_intersection": re.compile(r"\bintersection\b|\b at \b| and ", re.I),
    "narrative_mentions_entered_lane": re.compile(r"entered .*lane|into the .*lane|cut in", re.I),
    "narrative_mentions_crosswalk": re.compile(r"\bcrosswalk\b", re.I),
    "narrative_mentions_parking_garage": re.compile(r"parking garage", re.I),
}
FEATURE_DESCRIPTIONS = {
    "report_id": "Stable report identifier from the DMV PDF slug.",
    "manufacturer_std": "Standardized manufacturer/operator name.",
    "manufacturer_group_top10": "Top-operator bucket with remaining firms grouped as Other.",
    "accident_date_parsed": "Parsed accident date as ISO date string.",
    "accident_year": "Accident calendar year.",
    "accident_month": "Accident calendar month.",
    "accident_day": "Day of month.",
    "accident_weekday": "Weekday number where Monday=0.",
    "accident_weekday_name": "Weekday name derived from accident date.",
    "accident_quarter": "Calendar quarter.",
    "is_weekend": "Whether the accident occurred on Saturday or Sunday.",
    "accident_hour": "Hour extracted from accident time.",
    "accident_minute": "Minute extracted from accident time.",
    "hour_bin": "Coarse hour bin for time-of-day analysis.",
    "is_night": "Night-time indicator based on accident hour.",
    "is_morning_peak": "Morning peak-hour indicator.",
    "is_evening_peak": "Evening peak-hour indicator.",
    "mode_binary_autonomous": "1 if resolved mode is autonomous.",
    "mode_binary_conventional": "1 if resolved mode is conventional.",
    "mode_binary_ambiguous": "1 if resolved mode is ambiguous.",
    "vehicle1_damage_ordinal": "Ordinal damage severity for vehicle 1.",
    "vehicle1_damage_moderate_or_worse": "1 if vehicle 1 damage is moderate or major.",
    "vehicle1_damage_minor_or_worse": "1 if vehicle 1 damage is minor, moderate, or major.",
    "location_text": "Combined location address and city.",
    "location_city_std": "Standardized accident city.",
    "location_county_std": "Standardized accident county.",
    "location_is_intersection_like": "1 if the location text looks like an intersection reference.",
    "location_near_keyword": "1 if the location text contains 'near'.",
    "location_at_keyword": "1 if the location text contains ' at '.",
    "location_and_keyword": "1 if the location text contains ' and '.",
    "location_bay_area_flag": "1 if the collision occurred in common Bay Area cities/counties.",
    "location_los_angeles_flag": "1 if the collision occurred in Los Angeles.",
    "vru_any": "1 if the report suggests any vulnerable road user involvement.",
    "injury_text_signal": "Narrative-derived weak label: 1 injury mentioned, 0 explicit no injury, blank otherwise.",
    "narrative_char_count": "Character count of the narrative text.",
    "narrative_word_count": "Word count of the narrative text.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the engineered feature table from parsed California DMV collision reports."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Normalized collision CSV input.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Feature CSV output.")
    parser.add_argument("--dictionary", default=str(DEFAULT_DICTIONARY), help="Feature dictionary output.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE), help="Markdown profile output.")
    return parser.parse_args()


def normalize_company(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    canonical = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    rules = [
        ("waymo", "Waymo"),
        ("cruise", "Cruise"),
        ("zoox", "Zoox"),
        ("weride", "WeRide"),
        ("apple", "Apple"),
        ("aurora", "Aurora"),
        ("pony", "Pony.ai"),
        ("mercedes", "Mercedes-Benz"),
        ("nuro", "Nuro"),
        ("aimotive", "aiMotive"),
        ("motional", "Motional"),
        ("autox", "AutoX"),
        ("apollo", "Apollo"),
        ("baidu", "Baidu"),
        ("may mobility", "May Mobility"),
        ("navya", "Navya"),
        ("toyota", "Toyota"),
    ]
    for needle, label in rules:
        if needle in canonical:
            return label
    return text


def normalize_place(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text.isupper() else text


def split_multi_values(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def add_multi_value_flags(df: pd.DataFrame, source_column: str, prefix: str, descriptions: list[dict[str, str]]) -> pd.DataFrame:
    observed = sorted({item for value in df[source_column].fillna("") for item in split_multi_values(value)})
    new_columns: dict[str, pd.Series] = {}
    for item in observed:
        col = f"{prefix}__{item}"
        new_columns[col] = df[source_column].fillna("").apply(
            lambda value, token=item: int(token in split_multi_values(value))
        )
        descriptions.append(
            {
                "column_name": col,
                "source_column": source_column,
                "feature_type": "multi_hot",
                "description": f"Indicator that '{item}' was selected in {source_column}.",
            }
        )
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df


def add_pattern_flags(df: pd.DataFrame, descriptions: list[dict[str, str]]) -> None:
    narrative = df["narrative_text"].fillna("")
    for column_name, pattern in NARRATIVE_PATTERNS.items():
        df[column_name] = narrative.apply(lambda value, pat=pattern: int(bool(pat.search(value))))
        descriptions.append(
            {
                "column_name": column_name,
                "source_column": "narrative_text",
                "feature_type": "regex_flag",
                "description": f"Regex-derived narrative indicator using pattern: {pattern.pattern}",
            }
        )


def build_profile(df: pd.DataFrame) -> str:
    lines = [
        "# DMV Collision Feature Profile",
        "",
        f"- rows: {len(df)}",
        f"- columns: {len(df.columns)}",
        f"- unique_manufacturers_std: {df['manufacturer_std'].nunique(dropna=True)}",
        f"- autonomous_rows: {int(df['mode_binary_autonomous'].sum())}",
        f"- conventional_rows: {int(df['mode_binary_conventional'].sum())}",
        f"- ambiguous_mode_rows: {int(df['mode_binary_ambiguous'].sum())}",
        f"- vru_any_rows: {int(df['vru_any'].sum())}",
        "",
        "## Top Manufacturers",
        "",
    ]
    top_manu = df["manufacturer_std"].fillna("").value_counts().head(12)
    for name, count in top_manu.items():
        if name:
            lines.append(f"- {name}: {count}")
    lines.extend(["", "## Top Collision Types", ""])
    top_collision = df["collision_type_resolved_labels"].fillna("").value_counts().head(12)
    for name, count in top_collision.items():
        if name:
            lines.append(f"- {name}: {count}")
    lines.extend(["", "## Key Coverage", ""])
    for column in [
        "manufacturer_name",
        "accident_date",
        "accident_time",
        "vehicle1_make",
        "vehicle1_model",
        "narrative_text",
        "mode_resolved",
        "vehicle1_damage_level",
        "collision_type_resolved_labels",
    ]:
        lines.append(f"- {column}: {int(df[column].fillna('').astype(str).ne('').sum())}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    dictionary_path = Path(args.dictionary).expanduser().resolve()
    profile_path = Path(args.profile).expanduser().resolve()

    df = pd.read_csv(input_path)
    feature_descriptions: list[dict[str, str]] = []

    df["manufacturer_std"] = df["manufacturer_name"].fillna("").apply(normalize_company)
    top10 = set(df["manufacturer_std"].value_counts().head(10).index)
    df["manufacturer_group_top10"] = df["manufacturer_std"].apply(lambda value: value if value in top10 else ("Other" if value else ""))

    accident_date = pd.to_datetime(df["accident_date"], errors="coerce")
    df["accident_date_parsed"] = accident_date.dt.date.astype("string")
    df["accident_year"] = accident_date.dt.year.astype("Int64")
    df["accident_month"] = accident_date.dt.month.astype("Int64")
    df["accident_day"] = accident_date.dt.day.astype("Int64")
    df["accident_weekday"] = accident_date.dt.weekday.astype("Int64")
    df["accident_weekday_name"] = accident_date.dt.day_name().fillna("")
    df["accident_quarter"] = accident_date.dt.quarter.astype("Int64")
    df["is_weekend"] = accident_date.dt.weekday.isin([5, 6]).astype(int)

    time_parts = df["accident_time"].fillna("").str.extract(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})")
    df["accident_hour"] = pd.to_numeric(time_parts["hour"], errors="coerce").astype("Int64")
    df["accident_minute"] = pd.to_numeric(time_parts["minute"], errors="coerce").astype("Int64")

    def hour_bin(hour: object) -> str:
        if pd.isna(hour):
            return ""
        hour = int(hour)
        if 0 <= hour <= 5:
            return "overnight"
        if 6 <= hour <= 9:
            return "morning_peak"
        if 10 <= hour <= 15:
            return "midday"
        if 16 <= hour <= 19:
            return "evening_peak"
        return "night"

    df["hour_bin"] = df["accident_hour"].apply(hour_bin)
    df["is_night"] = df["accident_hour"].apply(lambda x: int(pd.notna(x) and (int(x) < 6 or int(x) >= 20)))
    df["is_morning_peak"] = df["hour_bin"].eq("morning_peak").astype(int)
    df["is_evening_peak"] = df["hour_bin"].eq("evening_peak").astype(int)

    df["mode_binary_autonomous"] = df["mode_resolved"].eq("autonomous").astype(int)
    df["mode_binary_conventional"] = df["mode_resolved"].eq("conventional").astype(int)
    df["mode_binary_ambiguous"] = df["mode_resolved"].eq("ambiguous").astype(int)

    df["vehicle1_damage_ordinal"] = df["vehicle1_damage_level"].map(DAMAGE_ORDINAL).astype("Int64")
    df["vehicle1_damage_moderate_or_worse"] = df["vehicle1_damage_level"].isin(["moderate", "major"]).astype(int)
    df["vehicle1_damage_minor_or_worse"] = df["vehicle1_damage_level"].isin(["minor", "moderate", "major"]).astype(int)

    df["location_city_std"] = df["accident_city"].fillna("").apply(normalize_place)
    df["location_county_std"] = df["accident_county"].fillna("").apply(normalize_place)
    df["location_text"] = (
        df["accident_location_address"].fillna("").astype(str).str.strip()
        + ", "
        + df["location_city_std"].fillna("").astype(str).str.strip()
    ).str.strip(", ")
    location_text_lower = df["location_text"].str.lower()
    df["location_near_keyword"] = location_text_lower.str.contains(r"\bnear\b", regex=True).astype(int)
    df["location_at_keyword"] = location_text_lower.str.contains(r"\bat\b", regex=True).astype(int)
    df["location_and_keyword"] = location_text_lower.str.contains(r"\band\b", regex=True).astype(int)
    df["location_is_intersection_like"] = (
        (df["location_near_keyword"] == 1)
        | (df["location_at_keyword"] == 1)
        | (df["location_and_keyword"] == 1)
    ).astype(int)
    bay_area_tokens = {
        "San Francisco",
        "San Jose",
        "Mountain View",
        "Sunnyvale",
        "Santa Clara",
        "Palo Alto",
        "Oakland",
        "Berkeley",
        "Fremont",
        "Daly City",
    }
    df["location_bay_area_flag"] = df["location_city_std"].isin(bay_area_tokens).astype(int)
    df["location_los_angeles_flag"] = df["location_city_std"].eq("Los Angeles").astype(int)

    add_pattern_flags(df, feature_descriptions)
    def is_one(column_name: str) -> pd.Series:
        values = df[column_name].astype("string")
        return values.eq("1") | values.eq("1.0")

    df["vru_any"] = (
        is_one("vehicle1_involved_pedestrian")
        | is_one("vehicle1_involved_bicyclist")
        | is_one("vehicle2_involved_pedestrian")
        | is_one("vehicle2_involved_bicyclist")
        | (df["collision_type_resolved_labels"].fillna("").str.contains("vehicle_pedestrian"))
        | (df["narrative_mentions_pedestrian"] == 1)
        | (df["narrative_mentions_bicyclist"] == 1)
        | (df["narrative_mentions_scooter"] == 1)
        | (df["narrative_mentions_motorcycle"] == 1)
    ).astype(int)
    df["narrative_char_count"] = df["narrative_text"].fillna("").str.len()
    df["narrative_word_count"] = df["narrative_text"].fillna("").str.split().str.len().fillna(0).astype(int)

    def injury_text_signal(row: pd.Series) -> object:
        pos = row["narrative_mentions_injury"] == 1
        neg = row["narrative_mentions_no_injury"] == 1
        if pos and not neg:
            return 1
        if neg and not pos:
            return 0
        if pos and neg:
            narrative = str(row["narrative_text"])
            if re.search(
                r"sought medical attention|transported to (?:a )?hospital|minor headache|neck stiffness|back pain|"
                r"neck pain|complained of bodily injury|reported injuries|reported injury",
                narrative,
                re.I,
            ):
                return 1
        return pd.NA

    df["injury_text_signal"] = df.apply(injury_text_signal, axis=1).astype("Int64")

    for source_column, prefix in MULTI_VALUE_COLUMNS.items():
        df = add_multi_value_flags(df, source_column, prefix, feature_descriptions)

    for column_name, description in FEATURE_DESCRIPTIONS.items():
        if column_name in df.columns:
            feature_descriptions.append(
                {
                    "column_name": column_name,
                    "source_column": "",
                    "feature_type": "derived",
                    "description": description,
                }
            )

    feature_dictionary = []
    for column in df.columns:
        string_values = df[column].astype("string").fillna("")
        non_empty = int(string_values.ne("").sum())
        examples = []
        for value in string_values:
            if value and value not in examples:
                examples.append(value.replace("\n", " | "))
            if len(examples) >= 3:
                break
        desc_row = next((row for row in feature_descriptions if row["column_name"] == column), None)
        feature_dictionary.append(
            {
                "column_name": column,
                "source_column": desc_row["source_column"] if desc_row else "",
                "feature_type": desc_row["feature_type"] if desc_row else "base",
                "non_empty_count": non_empty,
                "example_values": "; ".join(examples),
                "description": desc_row["description"] if desc_row else "Base column carried from normalized CSV.",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    pd.DataFrame(feature_dictionary).to_csv(dictionary_path, index=False)
    profile_path.write_text(build_profile(df), encoding="utf-8")

    print(f"Feature rows: {len(df)}")
    print(f"Feature columns: {len(df.columns)}")
    print(f"Wrote feature table to {output_path}")
    print(f"Wrote feature dictionary to {dictionary_path}")
    print(f"Wrote profile summary to {profile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
