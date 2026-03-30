#!/usr/bin/env python3
"""Parse California DMV autonomous-vehicle collision-report PDFs into CSV tables.

Outputs:
    1. A wide raw CSV with one row per report and one column per PDF widget.
    2. A normalized CSV with human-readable field names for common modeling fields.
    3. A long CSV with one row per widget for auditability.
    4. A variable-dictionary CSV summarizing coverage and semantics.

The parser uses PyMuPDF because these PDFs contain flattened form values that are
not reliably exposed through basic AcroForm extraction.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import fitz


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PACKAGE_ROOT / "data" / "public_source_manifest" / "ca_dmv_collision_report_manifest.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "data" / "processed"

REVISION_PATTERN = re.compile(r"OL 316 \(REV\. ([^)]+)\)")
PAGE3_PATTERN = re.compile(
    r"^(WEATHER|LIGHTING|ROADWAY|ROAD CONDITIONS|MOVEMENT|TYPE|OTHER)\s+(.+?)\s+([12]|YES|NO)$"
)

WEATHER_CODES = {
    "A": "clear",
    "B": "cloudy",
    "C": "raining",
    "D": "snowing",
    "E": "fog_or_visibility",
    "F": "other",
    "G": "wind",
}
LIGHTING_CODES = {
    "A": "daylight",
    "B": "dusk_dawn",
    "C": "dark_street_lights",
    "D": "dark_no_street_lights",
    "E": "dark_street_lights_not_functioning",
}
ROADWAY_CODES = {
    "A": "dry",
    "B": "wet",
    "C": "snowy_icy",
    "D": "slippery_muddy_oily",
}
ROAD_CONDITIONS_CODES = {
    "A": "holes_deep_rut",
    "B": "loose_material_on_roadway",
    "C": "obstruction_on_roadway",
    "D": "construction_repair_zone",
    "E": "reduced_roadway_width",
    "F": "flooded",
    "G": "other",
    "H": "no_unusual_conditions",
}
MOVEMENT_CODES = {
    "A": "stopped",
    "B": "proceeding_straight",
    "C": "ran_off_road",
    "D": "making_right_turn",
    "E": "making_left_turn",
    "F": "making_u_turn",
    "G": "backing",
    "H": "slowing_stopping",
    "I": "passing_other_vehicle",
    "J": "changing_lanes",
    "K": "parking_maneuver",
    "L": "entering_traffic",
    "M": "other_unsafe_turning",
    "N": "crossing_into_opposing_lane",
    "O": "parked",
    "P": "merging",
    "Q": "traveling_wrong_way",
    "R": "other",
}
TYPE_CODES = {
    "A": "head_on",
    "B": "side_swipe",
    "C": "rear_end",
    "D": "broadside",
    "E": "hit_object",
    "F": "overturned",
    "G": "vehicle_pedestrian",
    "H": "other",
}
OTHER_CODES = {
    "A": "cvc_sections_violated_cited",
    "B": "vision_obscurement",
    "C": "inattention",
    "D": "stop_go_traffic",
    "E": "entering_leaving_ramp",
    "F": "previous_collision",
    "G": "unfamiliar_with_road",
    "H": "defective_vehicle_equipment_cited",
    "I": "uninvolved_vehicle",
    "J": "other",
    "K": "none_apparent",
    "L": "runaway_vehicle",
}
DAMAGE_LEVEL_FIELDS = {
    "Unknown": "unknown",
    "None": "none",
    "minor": "minor",
    "Moderate": "moderate",
    "major": "major",
}
DAMAGE_ZONE_FIELDS = {
    "Left Rear 1": "left_rear_1",
    "Rear Bumper": "rear_bumper",
    "Right Rear 1": "right_rear_1",
    "Left Rear 2": "left_rear_2",
    "Left Rear 3": "left_rear_3",
    "Right Rear 2": "right_rear_2",
    "Right Rear 3": "right_rear_3",
    "Left Rear Passenger 1": "left_rear_passenger_1",
    "Left Rear Passenger 2": "left_rear_passenger_2",
    "Right Rear Passenger 1": "right_rear_passenger_1",
    "Right Rear Passenger 2": "right_rear_passenger_2",
    "Left Rear Passenger 3": "left_rear_passenger_3",
    "Left Rear Passenger 4": "left_rear_passenger_4",
    "Right Rear Passenger 3": "right_rear_passenger_3",
    "Right Rear Passenger 4": "right_rear_passenger_4",
    "Front Driver Side 1": "front_driver_side_1",
    "Front Driver Side 2": "front_driver_side_2",
    "Front Passenger Side 1": "front_passenger_side_1",
    "Front Passenger Side 2": "front_passenger_side_2",
    "Front Driver Side 3": "front_driver_side_3",
    "Front Driver Side 4": "front_driver_side_4",
    "Front Passenger Side 3": "front_passenger_side_3",
    "Front Passenger Side 4": "front_passenger_side_4",
    "Left Front Corner 1": "left_front_corner_1",
    "Left Front Corner 2": "left_front_corner_2",
    "Right Front Corner 1": "right_front_corner_1",
    "Right Front Corner 2": "right_front_corner_2",
    "Left Front Corner 3": "left_front_corner_3",
    "Front Bumper": "front_bumper",
    "Right Front Corner 3": "right_front_corner_3",
}
DERIVED_DESCRIPTIONS = {
    "report_id": "Stable report identifier derived from the PDF slug in the manifest.",
    "report_title": "DMV page title for the report entry.",
    "report_date_manifest": "Report date from the DMV page manifest.",
    "report_year_folder": "Year-based folder used for local storage.",
    "page_url": "Original DMV page URL serving the PDF.",
    "pdf_path": "Relative path to the mirrored PDF file.",
    "pdf_exists": "Whether the PDF file existed when parsing ran.",
    "page_count": "Number of pages in the PDF.",
    "ol316_revision": "Form revision parsed from the first page header.",
    "manufacturer_name": "Manufacturer name from Section 1.",
    "avt_number": "AVT permit number from Section 1.",
    "business_name": "Business name from Section 1.",
    "business_phone": "Combined business phone number from Section 1.",
    "manufacturer_street_address": "Manufacturer street address from Section 1.",
    "manufacturer_city": "Manufacturer city from Section 1.",
    "manufacturer_state": "Manufacturer state from Section 1.",
    "manufacturer_zip_code": "Manufacturer ZIP code from Section 1.",
    "accident_date": "Accident date from Section 2.",
    "accident_time": "Accident time from Section 2.",
    "accident_meridiem": "AM or PM marker from Section 2.",
    "vehicle1_year": "Testing vehicle model year.",
    "vehicle1_make": "Testing vehicle make.",
    "vehicle1_model": "Testing vehicle model.",
    "vehicle1_license_plate": "Testing vehicle license plate number.",
    "vehicle1_vin": "Testing vehicle VIN.",
    "vehicle1_registration_state": "State where vehicle 1 is registered.",
    "accident_location_address": "Accident address/location field.",
    "accident_city": "Accident city field.",
    "accident_county": "Accident county field.",
    "accident_state": "Accident state field.",
    "accident_zip_code": "Accident ZIP code field.",
    "vehicle1_moving": "Vehicle 1 marked as moving.",
    "vehicle1_stopped_in_traffic": "Vehicle 1 marked as stopped in traffic.",
    "vehicle1_involved_pedestrian": "Vehicle 1 involved a pedestrian.",
    "vehicle1_involved_bicyclist": "Vehicle 1 involved a bicyclist.",
    "vehicle1_involved_other": "Vehicle 1 involved some other party type.",
    "vehicle1_other_text": "Free-text description for vehicle 1 'Other' party type.",
    "number_of_vehicles_involved": "Number of vehicles involved in the collision.",
    "vehicle1_driver_name": "Vehicle 1 driver name field.",
    "vehicle1_driver_license_number": "Vehicle 1 driver license number field.",
    "vehicle1_driver_license_state": "Vehicle 1 driver license state field.",
    "vehicle1_driver_date_of_birth": "Vehicle 1 driver date of birth field.",
    "vehicle1_insurance_company": "Vehicle 1 insurance or surety company.",
    "vehicle1_policy_number": "Vehicle 1 insurance policy number.",
    "vehicle1_company_naic_number": "Vehicle 1 insurer NAIC number.",
    "vehicle1_policy_from": "Vehicle 1 insurance policy start date.",
    "vehicle1_policy_to": "Vehicle 1 insurance policy end date.",
    "vehicle1_damage_level": "Damage severity selected for vehicle 1.",
    "vehicle1_damage_zones": "Semicolon-separated selected vehicle 1 damage diagram regions.",
    "vehicle2_year": "Other party vehicle year from Section 3.",
    "vehicle2_model": "Other party vehicle model field from Section 3.",
    "vehicle2_license_plate": "Vehicle 2 license plate number.",
    "vehicle2_vin": "Vehicle 2 VIN.",
    "vehicle2_registration_state": "State where vehicle 2 is registered.",
    "vehicle2_moving": "Vehicle 2 marked as moving.",
    "vehicle2_stopped_in_traffic": "Vehicle 2 marked as stopped in traffic.",
    "vehicle2_involved_pedestrian": "Vehicle 2 involved a pedestrian.",
    "vehicle2_involved_bicyclist": "Vehicle 2 involved a bicyclist.",
    "vehicle2_involved_other": "Vehicle 2 involved some other party type.",
    "vehicle2_other_text": "Free-text description for vehicle 2 'Other' party type.",
    "vehicle2_driver_name": "Vehicle 2 driver name field.",
    "vehicle2_driver_license_number": "Vehicle 2 driver license number field.",
    "vehicle2_driver_license_state": "Vehicle 2 driver license state field.",
    "vehicle2_driver_date_of_birth": "Vehicle 2 driver date of birth field.",
    "vehicle2_insurance_company": "Vehicle 2 insurance or surety company.",
    "vehicle2_policy_number": "Vehicle 2 insurance policy number.",
    "vehicle2_company_naic_number": "Vehicle 2 insurer NAIC number.",
    "vehicle2_policy_from": "Vehicle 2 insurance policy start date.",
    "vehicle2_policy_to": "Vehicle 2 insurance policy end date.",
    "vehicle2_additional_information_attached": "Whether Section 3 indicates additional information attached.",
    "party1_name": "Section 4 first person or property-damage record name.",
    "party1_address": "Section 4 first record address.",
    "party1_city": "Section 4 first record city.",
    "party1_state": "Section 4 first record state.",
    "party1_zip_code": "Section 4 first record ZIP code.",
    "party1_injured": "Section 4 first record marked injured.",
    "party1_deceased": "Section 4 first record marked deceased.",
    "party1_driver": "Section 4 first record marked driver.",
    "party1_passenger": "Section 4 first record marked passenger.",
    "party1_bicyclist": "Section 4 first record marked bicyclist.",
    "party1_property": "Section 4 first record marked property.",
    "party2_name": "Section 4 second person or property-damage record name.",
    "party2_address": "Section 4 second record address.",
    "party2_city": "Section 4 second record city.",
    "party2_state": "Section 4 second record state.",
    "party2_zip_code": "Section 4 second record ZIP code.",
    "party2_injured": "Section 4 second record marked injured.",
    "party2_deceased": "Section 4 second record marked deceased.",
    "party2_driver": "Section 4 second record marked driver.",
    "party2_passenger": "Section 4 second record marked passenger.",
    "party2_bicyclist": "Section 4 second record marked bicyclist.",
    "party2_property": "Section 4 second record marked property.",
    "property_damage_description": "Section 4 property damage description.",
    "property_owner_name": "Property owner name from Section 4.",
    "property_owner_phone": "Combined property owner phone number.",
    "property_owner_address": "Property owner street address.",
    "property_owner_city": "Property owner city.",
    "property_owner_state": "Property owner state.",
    "property_owner_zip_code": "Property owner ZIP code.",
    "witness1_name": "First witness name.",
    "witness1_phone": "Combined first witness phone number.",
    "witness1_address": "First witness street address.",
    "witness1_city": "First witness city.",
    "witness1_state": "First witness state.",
    "witness1_zip_code": "First witness ZIP code.",
    "witness2_name": "Second witness name.",
    "witness2_phone": "Combined second witness phone number.",
    "witness2_address": "Second witness street address.",
    "witness2_city": "Second witness city.",
    "witness2_state": "Second witness state.",
    "witness2_zip_code": "Second witness ZIP code.",
    "section4_additional_information_attached": "Whether Section 4 indicates additional information attached.",
    "mode_autonomous": "Section 5 marked autonomous mode.",
    "mode_conventional": "Section 5 marked conventional mode.",
    "mode_from_form": "Driving mode inferred directly from Section 5 checkboxes.",
    "mode_from_narrative": "Driving mode heuristically inferred from the narrative text.",
    "mode_resolved": "Best-effort mode label preferring the form checkbox and falling back to the narrative.",
    "narrative_text": "Full free-text accident description from Section 5.",
    "section5_additional_information_attached": "Whether Section 5 indicates additional information attached.",
    "weather_vehicle1_labels": "Selected page-3 weather labels for vehicle 1.",
    "weather_vehicle2_labels": "Selected page-3 weather labels for vehicle 2.",
    "lighting_vehicle1_labels": "Selected page-3 lighting labels for vehicle 1.",
    "lighting_vehicle2_labels": "Selected page-3 lighting labels for vehicle 2.",
    "roadway_surface_vehicle1_labels": "Selected roadway surface labels for vehicle 1.",
    "roadway_surface_vehicle2_labels": "Selected roadway surface labels for vehicle 2.",
    "movement_vehicle1_labels": "Selected movement-preceding-collision labels for vehicle 1.",
    "movement_vehicle2_labels": "Selected movement-preceding-collision labels for vehicle 2.",
    "road_conditions_vehicle1_labels": "Selected roadway-condition labels for vehicle 1.",
    "road_conditions_vehicle2_labels": "Selected roadway-condition labels for vehicle 2.",
    "collision_type_vehicle1_codes": "Selected collision-type codes for vehicle 1.",
    "collision_type_vehicle1_labels": "Selected collision-type labels for vehicle 1.",
    "collision_type_vehicle2_codes": "Selected collision-type codes for vehicle 2.",
    "collision_type_vehicle2_labels": "Selected collision-type labels for vehicle 2.",
    "collision_type_resolved_codes": "Best-effort collision type codes after combining vehicle 1 and vehicle 2 selections.",
    "collision_type_resolved_labels": "Best-effort collision type labels after combining vehicle 1 and vehicle 2 selections.",
    "other_associated_factor_labels": "Selected page-3 other associated factor labels.",
    "other_cvc_sections_violated_cited": "Page-3 citation flag for CVC sections violated.",
    "other_defective_vehicle_equipment_cited": "Page-3 citation flag for defective vehicle equipment.",
    "certifier_name_title": "Program director or authorized representative printed name and title.",
    "certifier_phone": "Combined certifier phone number.",
    "date_signed": "Certification signature date.",
}


@dataclass
class WidgetRecord:
    page_number: int
    field_name: str
    field_type: str
    value: str
    checked: Optional[bool]
    rect: tuple[float, float, float, float]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def normalize_identifier(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "field"


def unique_column_map(raw_names: Iterable[str], prefix: str) -> dict[str, str]:
    counts: Counter[str] = Counter()
    mapping: dict[str, str] = {}
    for raw_name in sorted(set(raw_names)):
        base = f"{prefix}_{normalize_identifier(raw_name)}"
        counts[base] += 1
        mapping[raw_name] = base if counts[base] == 1 else f"{base}_{counts[base]}"
    return mapping


def checkbox_selected(value: str) -> bool:
    return value not in {"", "Off", "No", "0"}


def combine_parts(*parts: str) -> str:
    pieces = [clean_text(part) for part in parts if clean_text(part)]
    return " ".join(pieces)


def semicolon_join(values: Iterable[str]) -> str:
    items = [value for value in values if value]
    return "; ".join(items)


def parse_revision(doc: fitz.Document) -> str:
    if len(doc) == 0:
        return ""
    text = doc[0].get_text("text")
    match = REVISION_PATTERN.search(text)
    return match.group(1) if match else ""


def read_widgets(pdf_path: Path) -> tuple[list[WidgetRecord], str, int]:
    doc = fitz.open(pdf_path)
    revision = parse_revision(doc)
    records: list[WidgetRecord] = []

    for page_number, page in enumerate(doc, start=1):
        widgets = list(page.widgets() or [])
        for widget in widgets:
            field_name = clean_text(widget.field_name)
            field_type = widget.field_type_string or ""
            if not field_name or field_type == "Button":
                continue

            raw_value = clean_text(widget.field_value)
            checked: Optional[bool] = None
            value = raw_value
            if field_type == "CheckBox":
                checked = checkbox_selected(raw_value)
                value = "1" if checked else "0"

            rect = tuple(round(v, 3) for v in widget.rect)
            records.append(
                WidgetRecord(
                    page_number=page_number,
                    field_name=field_name,
                    field_type=field_type,
                    value=value,
                    checked=checked,
                    rect=rect,
                )
            )

    return records, revision, len(doc)


def widget_map(widget_records: list[WidgetRecord]) -> dict[str, WidgetRecord]:
    mapping: dict[str, WidgetRecord] = {}
    for record in widget_records:
        existing = mapping.get(record.field_name)
        if existing is None:
            mapping[record.field_name] = record
            continue
        if not existing.value and record.value:
            mapping[record.field_name] = record
    return mapping


def text_value(mapping: dict[str, WidgetRecord], name: str) -> str:
    record = mapping.get(name)
    return record.value if record and record.field_type != "CheckBox" else ""


def bool_value(mapping: dict[str, WidgetRecord], name: str) -> bool:
    record = mapping.get(name)
    return bool(record and record.checked)


def page3_code_lists(mapping: dict[str, WidgetRecord], group: str, code_map: dict[str, str]) -> tuple[str, str, str, str]:
    vehicle1_codes: list[str] = []
    vehicle1_labels: list[str] = []
    vehicle2_codes: list[str] = []
    vehicle2_labels: list[str] = []

    for field_name, record in mapping.items():
        if record.field_type != "CheckBox" or not record.checked:
            continue
        match = PAGE3_PATTERN.match(field_name)
        if not match:
            continue
        field_group, code, suffix = match.groups()
        if field_group != group:
            continue
        if suffix == "1":
            vehicle1_codes.append(code)
            vehicle1_labels.append(code_map.get(code, code))
        elif suffix == "2":
            vehicle2_codes.append(code)
            vehicle2_labels.append(code_map.get(code, code))

    return (
        semicolon_join(vehicle1_codes),
        semicolon_join(vehicle1_labels),
        semicolon_join(vehicle2_codes),
        semicolon_join(vehicle2_labels),
    )


def other_factor_fields(mapping: dict[str, WidgetRecord]) -> tuple[str, str, str]:
    labels: list[str] = []
    cvc_cited = ""
    defective_cited = ""

    for field_name, record in mapping.items():
        if record.field_type != "CheckBox" or not record.checked:
            continue
        match = PAGE3_PATTERN.match(field_name)
        if not match:
            continue
        field_group, code, suffix = match.groups()
        if field_group != "OTHER":
            continue
        if code == "A" and suffix in {"YES", "NO"}:
            cvc_cited = suffix.lower()
            continue
        if code == "H" and suffix in {"YES", "NO"}:
            defective_cited = suffix.lower()
            continue
        labels.append(OTHER_CODES.get(code, code))

    return semicolon_join(labels), cvc_cited, defective_cited


def infer_mode_from_narrative(narrative_text: str) -> str:
    text = narrative_text.casefold()
    has_autonomous = any(
        phrase in text
        for phrase in (
            "autonomous mode",
            "operating in autonomous mode",
            "ads was engaged",
            "ads engaged",
            "driverless mode",
        )
    )
    has_conventional = any(
        phrase in text
        for phrase in (
            "conventional mode",
            "manual mode",
            "operating in conventional mode",
            "manual driving mode",
        )
    )
    if has_autonomous and not has_conventional:
        return "autonomous"
    if has_conventional and not has_autonomous:
        return "conventional"
    if has_autonomous and has_conventional:
        return "ambiguous"
    return ""


def resolved_collision_type(
    vehicle1_codes: str,
    vehicle1_labels: str,
    vehicle2_codes: str,
    vehicle2_labels: str,
) -> tuple[str, str]:
    codes: list[str] = []
    labels: list[str] = []
    for value in (vehicle1_codes, vehicle2_codes):
        for item in [part.strip() for part in value.split(";") if part.strip()]:
            if item not in codes:
                codes.append(item)
    for value in (vehicle1_labels, vehicle2_labels):
        for item in [part.strip() for part in value.split(";") if part.strip()]:
            if item not in labels:
                labels.append(item)
    return semicolon_join(codes), semicolon_join(labels)


def derive_row(manifest_row: dict[str, str], mapping: dict[str, WidgetRecord], revision: str, page_count: int) -> dict[str, str]:
    weather_v1_codes, weather_v1_labels, weather_v2_codes, weather_v2_labels = page3_code_lists(
        mapping, "WEATHER", WEATHER_CODES
    )
    lighting_v1_codes, lighting_v1_labels, lighting_v2_codes, lighting_v2_labels = page3_code_lists(
        mapping, "LIGHTING", LIGHTING_CODES
    )
    roadway_v1_codes, roadway_v1_labels, roadway_v2_codes, roadway_v2_labels = page3_code_lists(
        mapping, "ROADWAY", ROADWAY_CODES
    )
    movement_v1_codes, movement_v1_labels, movement_v2_codes, movement_v2_labels = page3_code_lists(
        mapping, "MOVEMENT", MOVEMENT_CODES
    )
    road_conditions_v1_codes, road_conditions_v1_labels, road_conditions_v2_codes, road_conditions_v2_labels = page3_code_lists(
        mapping, "ROAD CONDITIONS", ROAD_CONDITIONS_CODES
    )
    collision_type_v1_codes, collision_type_v1_labels, collision_type_v2_codes, collision_type_v2_labels = page3_code_lists(
        mapping, "TYPE", TYPE_CODES
    )
    other_labels, cvc_cited, defective_equipment_cited = other_factor_fields(mapping)
    narrative_text = text_value(mapping, "ADDRESS_2.1.0.1")
    mode_from_form = (
        "autonomous"
        if bool_value(mapping, "Autonomous Mode")
        else ("conventional" if bool_value(mapping, "Conventional Mode") else "")
    )
    mode_from_narrative = infer_mode_from_narrative(narrative_text)
    mode_resolved = mode_from_form or mode_from_narrative
    collision_type_resolved_codes, collision_type_resolved_labels = resolved_collision_type(
        collision_type_v1_codes,
        collision_type_v1_labels,
        collision_type_v2_codes,
        collision_type_v2_labels,
    )

    selected_damage_level = next(
        (label for raw_name, label in DAMAGE_LEVEL_FIELDS.items() if bool_value(mapping, raw_name)),
        "",
    )
    selected_damage_zones = semicolon_join(
        label for raw_name, label in DAMAGE_ZONE_FIELDS.items() if bool_value(mapping, raw_name)
    )

    row = {
        "report_id": manifest_row["slug"],
        "report_title": manifest_row["title"],
        "report_date_manifest": manifest_row["report_date"],
        "report_year_folder": manifest_row["year_folder"],
        "page_url": manifest_row["page_url"],
        "pdf_path": manifest_row["local_path"],
        "pdf_exists": "1",
        "page_count": str(page_count),
        "ol316_revision": revision,
        "manufacturer_name": text_value(mapping, "MANufACTuRERS NAME"),
        "avt_number": text_value(mapping, "AVT NuMBER_2"),
        "business_name": text_value(mapping, "BuSINESS NAME"),
        "business_phone": combine_parts(
            text_value(mapping, "TELEPhONE NuMBER"),
            text_value(mapping, "TELEPhONE NuMBER.0"),
            text_value(mapping, "TELEPhONE NuMBER.1"),
        ),
        "manufacturer_street_address": text_value(mapping, "DRIVERS fuLL NAME First Middle last.1"),
        "manufacturer_city": text_value(mapping, "DRIVER LICENSE NuMBER.1"),
        "manufacturer_state": text_value(mapping, "STATE.1"),
        "manufacturer_zip_code": text_value(mapping, "DATE Of BIRTh.1"),
        "accident_date": text_value(mapping, "DATE Of ACCIDENT"),
        "accident_time": text_value(mapping, "Time of Accident"),
        "accident_meridiem": "AM" if bool_value(mapping, "AM") else ("PM" if bool_value(mapping, "PM") else ""),
        "vehicle1_year": text_value(mapping, "VEhICLE YEAR"),
        "vehicle1_make": text_value(mapping, "MAkE"),
        "vehicle1_model": text_value(mapping, "MODEL"),
        "vehicle1_license_plate": text_value(mapping, "LICENSE PLATE NuMBER"),
        "vehicle1_vin": text_value(mapping, "VEhICLE IDENTIfICATION NuMBER"),
        "vehicle1_registration_state": text_value(mapping, "STATE VEhICLE IS REGISTERED IN"),
        "accident_location_address": text_value(mapping, "section 2  accident infoRmation.0"),
        "accident_city": text_value(mapping, "section 2  accident infoRmation.1.0"),
        "accident_county": text_value(mapping, "section 2  accident infoRmation.1.1.0"),
        "accident_state": text_value(mapping, "section 2  accident infoRmation.1.1.1.0"),
        "accident_zip_code": text_value(mapping, "section 2  accident infoRmation.1.1.1.1"),
        "vehicle1_moving": "1" if bool_value(mapping, "Moving") else "0",
        "vehicle1_stopped_in_traffic": "1" if bool_value(mapping, "Stopped in Traffic") else "0",
        "vehicle1_involved_pedestrian": "1" if bool_value(mapping, "Pedestrian") else "0",
        "vehicle1_involved_bicyclist": "1" if bool_value(mapping, "Bicyclist") else "0",
        "vehicle1_involved_other": "1" if bool_value(mapping, "undefined") else "0",
        "vehicle1_other_text": text_value(mapping, "Other"),
        "number_of_vehicles_involved": text_value(mapping, "NuMBER Of VEhICLES INVOLVED"),
        "vehicle1_driver_name": text_value(mapping, "DRIVERS fuLL NAME First Middle last.0"),
        "vehicle1_driver_license_number": text_value(mapping, "DRIVER LICENSE NuMBER.0"),
        "vehicle1_driver_license_state": text_value(mapping, "STATE.0"),
        "vehicle1_driver_date_of_birth": text_value(mapping, "DATE Of BIRTh.0"),
        "vehicle1_insurance_company": text_value(
            mapping, "INSuRANCE COMPANY NAME OR SuRETY COMPANY AT TIME Of ACCIDENT"
        ),
        "vehicle1_policy_number": text_value(mapping, "POLICY NuMBER"),
        "vehicle1_company_naic_number": text_value(mapping, "COMPANY NAIC NuMBER"),
        "vehicle1_policy_from": text_value(mapping, "fROM"),
        "vehicle1_policy_to": text_value(mapping, "TO"),
        "vehicle1_damage_level": selected_damage_level,
        "vehicle1_damage_zones": selected_damage_zones,
        "vehicle2_year": text_value(mapping, "VEhICLE YEAR_2"),
        "vehicle2_model": text_value(mapping, "MODEL_2"),
        "vehicle2_license_plate": text_value(mapping, "LICENSE PLATE NuMBER_2"),
        "vehicle2_vin": text_value(mapping, "VEhICLE IDENTIfICATION NuMBER_2"),
        "vehicle2_registration_state": text_value(mapping, "STATE VEhICLE IS REGISTERED IN_2"),
        "vehicle2_moving": "1" if bool_value(mapping, "Moving_2") else "0",
        "vehicle2_stopped_in_traffic": "1" if bool_value(mapping, "Stopped in Traffic_2") else "0",
        "vehicle2_involved_pedestrian": "1" if bool_value(mapping, "Pedestrian_2") else "0",
        "vehicle2_involved_bicyclist": "1" if bool_value(mapping, "Bicyclist_2") else "0",
        "vehicle2_involved_other": "1" if bool_value(mapping, "undefined_2") else "0",
        "vehicle2_other_text": text_value(mapping, "Other_2"),
        "vehicle2_driver_name": text_value(mapping, "DRIVERS fuLL NAME First Middle last_2"),
        "vehicle2_driver_license_number": text_value(mapping, "DRIVER LICENSE NuMBER_2"),
        "vehicle2_driver_license_state": text_value(mapping, "STATE_2"),
        "vehicle2_driver_date_of_birth": text_value(mapping, "DATE Of BIRTh_2"),
        "vehicle2_insurance_company": text_value(
            mapping, "INSuRANCE COMPANY NAME OR SuRETY COMPANY AT TIME Of ACCIDENT_2"
        ),
        "vehicle2_policy_number": text_value(mapping, "POLICY NuMBER_2"),
        "vehicle2_company_naic_number": text_value(mapping, "COMPANY NAIC NuMBER_2"),
        "vehicle2_policy_from": text_value(mapping, "fROM_2"),
        "vehicle2_policy_to": text_value(mapping, "TO_2"),
        "vehicle2_additional_information_attached": "1"
        if bool_value(mapping, "additional information attached")
        else "0",
        "party1_name": text_value(mapping, "NAME First Middle last"),
        "party1_address": text_value(mapping, "ADDRESS"),
        "party1_city": text_value(mapping, "CITY"),
        "party1_state": text_value(mapping, "STATE_3"),
        "party1_zip_code": text_value(mapping, "ZIP CODE"),
        "party1_injured": "1" if bool_value(mapping, "Injured") else "0",
        "party1_deceased": "1" if bool_value(mapping, "Deceased") else "0",
        "party1_driver": "1" if bool_value(mapping, "Driver") else "0",
        "party1_passenger": "1" if bool_value(mapping, "Passenger") else "0",
        "party1_bicyclist": "1" if bool_value(mapping, "Bicyclist_3") else "0",
        "party1_property": "1" if bool_value(mapping, "Proper ty") else "0",
        "party2_name": text_value(mapping, "NAME First Middle last_2"),
        "party2_address": text_value(mapping, "ADDRESS_2.0"),
        "party2_city": text_value(mapping, "CITY_2.0"),
        "party2_state": text_value(mapping, "STATE_4.0"),
        "party2_zip_code": text_value(mapping, "ZIP CODE_2.0"),
        "party2_injured": "1" if bool_value(mapping, "Injured_2") else "0",
        "party2_deceased": "1" if bool_value(mapping, "Deceased_2") else "0",
        "party2_driver": "1" if bool_value(mapping, "Driver_2") else "0",
        "party2_passenger": "1" if bool_value(mapping, "Passenger_2") else "0",
        "party2_bicyclist": "1" if bool_value(mapping, "Bicyclist_4") else "0",
        "party2_property": "1" if bool_value(mapping, "Proper ty_2") else "0",
        "property_damage_description": text_value(mapping, "PROPERTY DAMAGE"),
        "property_owner_name": text_value(mapping, "PROPERTY OWNERS NAME"),
        "property_owner_phone": combine_parts(
            text_value(mapping, "TELEPhONE NuMBER_2.0"),
            text_value(mapping, "TELEPhONE NuMBER_2.1.0"),
        ),
        "property_owner_address": text_value(mapping, "ADDRESS_2.1.1.0"),
        "property_owner_city": text_value(mapping, "CITY_2.1.1.0"),
        "property_owner_state": text_value(mapping, "STATE_4.1.1.0"),
        "property_owner_zip_code": text_value(mapping, "ZIP CODE_2.1.1.0"),
        "witness1_name": text_value(mapping, "WITNESS NAME"),
        "witness1_phone": combine_parts(
            text_value(mapping, "TELEPhONE NuMBER_3"),
            text_value(mapping, "TELEPhONE NuMBER_2.1.1"),
        ),
        "witness1_address": text_value(mapping, "ADDRESS_2.1.1.1"),
        "witness1_city": text_value(mapping, "CITY_2.1.1.1"),
        "witness1_state": text_value(mapping, "STATE_4.1.1.1"),
        "witness1_zip_code": text_value(mapping, "ZIP CODE_2.1.1.1"),
        "witness2_name": text_value(mapping, "WITNESS NAME_2"),
        "witness2_phone": combine_parts(
            text_value(mapping, "TELEPhONE NuMBER_4.0"),
            text_value(mapping, "TELEPhONE NuMBER_2.1.2.0"),
        ),
        "witness2_address": text_value(mapping, "ADDRESS_2.1.0.0"),
        "witness2_city": text_value(mapping, "CITY_2.1.0"),
        "witness2_state": text_value(mapping, "STATE_4.1.0"),
        "witness2_zip_code": text_value(mapping, "ZIP CODE_2.1.0"),
        "section4_additional_information_attached": "1"
        if bool_value(mapping, "additional information attached_2")
        else "0",
        "mode_autonomous": "1" if bool_value(mapping, "Autonomous Mode") else "0",
        "mode_conventional": "1" if bool_value(mapping, "Conventional Mode") else "0",
        "mode_from_form": mode_from_form,
        "mode_from_narrative": mode_from_narrative,
        "mode_resolved": mode_resolved,
        "narrative_text": narrative_text,
        "section5_additional_information_attached": "1"
        if bool_value(mapping, "additional information attached_3")
        else "0",
        "weather_vehicle1_codes": weather_v1_codes,
        "weather_vehicle1_labels": weather_v1_labels,
        "weather_vehicle2_codes": weather_v2_codes,
        "weather_vehicle2_labels": weather_v2_labels,
        "lighting_vehicle1_codes": lighting_v1_codes,
        "lighting_vehicle1_labels": lighting_v1_labels,
        "lighting_vehicle2_codes": lighting_v2_codes,
        "lighting_vehicle2_labels": lighting_v2_labels,
        "roadway_surface_vehicle1_codes": roadway_v1_codes,
        "roadway_surface_vehicle1_labels": roadway_v1_labels,
        "roadway_surface_vehicle2_codes": roadway_v2_codes,
        "roadway_surface_vehicle2_labels": roadway_v2_labels,
        "movement_vehicle1_codes": movement_v1_codes,
        "movement_vehicle1_labels": movement_v1_labels,
        "movement_vehicle2_codes": movement_v2_codes,
        "movement_vehicle2_labels": movement_v2_labels,
        "road_conditions_vehicle1_codes": road_conditions_v1_codes,
        "road_conditions_vehicle1_labels": road_conditions_v1_labels,
        "road_conditions_vehicle2_codes": road_conditions_v2_codes,
        "road_conditions_vehicle2_labels": road_conditions_v2_labels,
        "collision_type_vehicle1_codes": collision_type_v1_codes,
        "collision_type_vehicle1_labels": collision_type_v1_labels,
        "collision_type_vehicle2_codes": collision_type_v2_codes,
        "collision_type_vehicle2_labels": collision_type_v2_labels,
        "collision_type_resolved_codes": collision_type_resolved_codes,
        "collision_type_resolved_labels": collision_type_resolved_labels,
        "other_associated_factor_labels": other_labels,
        "other_cvc_sections_violated_cited": cvc_cited,
        "other_defective_vehicle_equipment_cited": defective_equipment_cited,
        "certifier_name_title": text_value(
            mapping, "PROGRAM DIRECTORAuThORIZED REPRESENTATIVE PRINTED NAME AND TITLE"
        ),
        "certifier_phone": combine_parts(
            text_value(mapping, "TELEPhONE NuMBER_4.1"),
            text_value(mapping, "TELEPhONE NuMBER_2.1.2.1"),
        ),
        "date_signed": text_value(mapping, "DATE SIGNED"),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_variable_dictionary(
    manifest_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    raw_field_map: dict[str, str],
    raw_field_meta: dict[str, dict[str, object]],
    normalized_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    dictionary_rows: list[dict[str, str]] = []

    manifest_descriptions = {
        "title": "Title text shown on the DMV collision reports page.",
        "manufacturer": "Manufacturer parsed from the DMV page link text.",
        "report_date": "Report date parsed from the DMV page link text.",
        "year_folder": "Year folder used when downloading PDFs.",
        "page_url": "Original DMV page URL for the PDF.",
        "slug": "Slug derived from the DMV page URL.",
        "local_path": "Relative path to the mirrored PDF file.",
    }
    for column_name, description in manifest_descriptions.items():
        coverage = sum(1 for row in manifest_rows if row.get(column_name))
        dictionary_rows.append(
            {
                "column_name": column_name,
                "layer": "manifest",
                "raw_field_name": "",
                "field_type": "text",
                "coverage_count": str(coverage),
                "non_empty_count": str(coverage),
                "positive_count": "",
                "example_values": semicolon_join(
                    row[column_name] for row in manifest_rows[:3] if row.get(column_name)
                ),
                "description": description,
            }
        )

    for raw_name, column_name in sorted(raw_field_map.items(), key=lambda item: item[1]):
        meta = raw_field_meta[raw_name]
        dictionary_rows.append(
            {
                "column_name": column_name,
                "layer": "raw_widget",
                "raw_field_name": raw_name,
                "field_type": str(meta["field_type"]),
                "coverage_count": str(meta["coverage_count"]),
                "non_empty_count": str(meta["non_empty_count"]),
                "positive_count": str(meta["positive_count"]),
                "example_values": "; ".join(meta["examples"]),
                "description": f"Raw PDF widget '{raw_name}' extracted from the OL 316 form.",
            }
        )

    normalized_columns = list(normalized_rows[0].keys()) if normalized_rows else []
    for column_name in normalized_columns:
        non_empty_count = sum(1 for row in normalized_rows if row.get(column_name))
        positive_count = sum(
            1
            for row in normalized_rows
            if row.get(column_name) in {"1", "yes", "true", "AM", "PM"}
        )
        examples = []
        for row in normalized_rows:
            value = row.get(column_name, "")
            if value and value not in examples:
                examples.append(value.replace("\n", " | "))
            if len(examples) >= 3:
                break
        dictionary_rows.append(
            {
                "column_name": column_name,
                "layer": "normalized",
                "raw_field_name": "",
                "field_type": "derived",
                "coverage_count": str(len(normalized_rows)),
                "non_empty_count": str(non_empty_count),
                "positive_count": str(positive_count),
                "example_values": "; ".join(examples),
                "description": DERIVED_DESCRIPTIONS.get(column_name, "Normalized derived field."),
            }
        )

    return dictionary_rows


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse California DMV autonomous-vehicle collision report PDFs into raw, "
            "normalized, and long-form CSV files."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the DMV collision report manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where parsed CSV outputs will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only parse the first N manifest rows for testing.",
    )
    parser.add_argument(
        "--year",
        action="append",
        type=int,
        help="Restrict parsing to one or more report years.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    if args.year:
        year_set = {str(year) for year in args.year}
        manifest_rows = [row for row in manifest_rows if row.get("year_folder") in year_set]
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    parsed_reports: list[tuple[dict[str, str], list[WidgetRecord], dict[str, WidgetRecord], str, int]] = []
    raw_field_names: set[str] = set()
    raw_field_meta: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "field_type": "",
            "coverage_count": 0,
            "non_empty_count": 0,
            "positive_count": 0,
            "examples": [],
        }
    )

    for index, manifest_row in enumerate(manifest_rows, start=1):
        pdf_path = Path(manifest_row["local_path"])
        if not pdf_path.exists():
            print(f"[{index}/{len(manifest_rows)}] missing PDF: {pdf_path}", file=sys.stderr)
            continue

        widget_records, revision, page_count = read_widgets(pdf_path)
        mapping = widget_map(widget_records)
        parsed_reports.append((manifest_row, widget_records, mapping, revision, page_count))

        for field_name, record in mapping.items():
            raw_field_names.add(field_name)
            meta = raw_field_meta[field_name]
            meta["field_type"] = record.field_type
            meta["coverage_count"] = int(meta["coverage_count"]) + 1
            if record.value:
                meta["non_empty_count"] = int(meta["non_empty_count"]) + 1
            if record.checked:
                meta["positive_count"] = int(meta["positive_count"]) + 1
            if record.value and record.value not in meta["examples"] and len(meta["examples"]) < 3:
                meta["examples"].append(record.value.replace("\n", " | "))

    raw_field_map = unique_column_map(raw_field_names, prefix="raw")

    raw_wide_rows: list[dict[str, str]] = []
    normalized_rows: list[dict[str, str]] = []
    long_rows: list[dict[str, str]] = []

    for manifest_row, widget_records, mapping, revision, page_count in parsed_reports:
        raw_row = {
            "report_id": manifest_row["slug"],
            "title": manifest_row["title"],
            "manufacturer": manifest_row["manufacturer"],
            "report_date": manifest_row["report_date"],
            "year_folder": manifest_row["year_folder"],
            "page_url": manifest_row["page_url"],
            "slug": manifest_row["slug"],
            "local_path": manifest_row["local_path"],
            "ol316_revision": revision,
            "page_count": str(page_count),
        }

        for field_name, column_name in raw_field_map.items():
            raw_row[column_name] = ""
        for field_name, record in mapping.items():
            raw_row[raw_field_map[field_name]] = record.value

        raw_wide_rows.append(raw_row)
        normalized_rows.append(derive_row(manifest_row, mapping, revision, page_count))

        for record in widget_records:
            long_rows.append(
                {
                    "report_id": manifest_row["slug"],
                    "report_date": manifest_row["report_date"],
                    "year_folder": manifest_row["year_folder"],
                    "page_number": str(record.page_number),
                    "field_name_raw": record.field_name,
                    "field_name_normalized": raw_field_map[record.field_name],
                    "field_type": record.field_type,
                    "value": record.value,
                    "checked": ""
                    if record.checked is None
                    else ("1" if record.checked else "0"),
                    "rect_x0": str(record.rect[0]),
                    "rect_y0": str(record.rect[1]),
                    "rect_x1": str(record.rect[2]),
                    "rect_y1": str(record.rect[3]),
                }
            )

    variable_dictionary_rows = build_variable_dictionary(
        manifest_rows=manifest_rows,
        raw_rows=raw_wide_rows,
        raw_field_map=raw_field_map,
        raw_field_meta=raw_field_meta,
        normalized_rows=normalized_rows,
    )

    raw_fieldnames = [
        "report_id",
        "title",
        "manufacturer",
        "report_date",
        "year_folder",
        "page_url",
        "slug",
        "local_path",
        "ol316_revision",
        "page_count",
        *[raw_field_map[field_name] for field_name in sorted(raw_field_names)],
    ]
    normalized_fieldnames = list(normalized_rows[0].keys()) if normalized_rows else []
    long_fieldnames = [
        "report_id",
        "report_date",
        "year_folder",
        "page_number",
        "field_name_raw",
        "field_name_normalized",
        "field_type",
        "value",
        "checked",
        "rect_x0",
        "rect_y0",
        "rect_x1",
        "rect_y1",
    ]
    dictionary_fieldnames = [
        "column_name",
        "layer",
        "raw_field_name",
        "field_type",
        "coverage_count",
        "non_empty_count",
        "positive_count",
        "example_values",
        "description",
    ]

    write_csv(output_dir / "dmv_collision_reports_raw_wide.csv", raw_wide_rows, raw_fieldnames)
    write_csv(output_dir / "dmv_collision_reports_normalized.csv", normalized_rows, normalized_fieldnames)
    write_csv(output_dir / "dmv_collision_reports_widgets_long.csv", long_rows, long_fieldnames)
    write_csv(output_dir / "dmv_collision_reports_variable_dictionary.csv", variable_dictionary_rows, dictionary_fieldnames)

    print(f"Parsed reports: {len(parsed_reports)}")
    print(f"Raw widget fields: {len(raw_field_names)}")
    print(f"Outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
