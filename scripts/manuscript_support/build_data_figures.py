from __future__ import annotations

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PACKAGE_ROOT / "data" / "processed" / "engineered_features.csv"
RAW_CSV = PACKAGE_ROOT / "data" / "processed" / "parsed_reports_raw_wide.csv"
NORM_CSV = PACKAGE_ROOT / "data" / "processed" / "parsed_reports_normalized.csv"
WIDGETS_CSV = PACKAGE_ROOT / "data" / "processed" / "parsed_widgets_long.csv"
FIGURES_DIR = PACKAGE_ROOT / "manuscript_assets" / "figures"
TABLES_DIR = PACKAGE_ROOT / "manuscript_assets" / "tables"
OLD_XLSX = PACKAGE_ROOT / "external" / "legacy_ca_dmv_structured_table.xlsx"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
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


def load_data() -> tuple[pd.DataFrame, dict[str, int]]:
    feat = pd.read_csv(FEATURES_CSV)
    raw_cols = len(pd.read_csv(RAW_CSV, nrows=2).columns)
    norm_cols = len(pd.read_csv(NORM_CSV, nrows=2).columns)
    feat_cols = len(pd.read_csv(FEATURES_CSV, nrows=2).columns)
    widget_rows = len(pd.read_csv(WIDGETS_CSV, usecols=[0]))

    stats = {
        "public_pdfs": int(len(feat)),
        "raw_cols": raw_cols,
        "norm_cols": norm_cols,
        "feat_cols": feat_cols,
        "widget_rows": widget_rows,
        "mode_nonambiguous": int(feat["mode_resolved"].isin(["autonomous", "conventional"]).sum()),
        "injury_labeled": int(feat["injury_text_signal"].notna().sum()),
    }
    final_sample = feat[feat["mode_resolved"].isin(["autonomous", "conventional"])].copy()
    final_sample = final_sample[final_sample["injury_text_signal"].notna()].copy()
    stats["final_sample"] = int(len(final_sample))
    stats["final_positives"] = int(final_sample["injury_text_signal"].sum())
    stats["mode_autonomous"] = int((final_sample["mode_resolved"] == "autonomous").sum())
    stats["mode_conventional"] = int((final_sample["mode_resolved"] == "conventional").sum())
    return feat, stats


def build_validation_summary(feat: pd.DataFrame) -> dict[str, object]:
    old = pd.read_excel(OLD_XLSX, sheet_name="Crash Data")
    old["date_parsed"] = pd.to_datetime(old["Date"], dayfirst=True, errors="coerce").dt.date
    old["company_norm"] = old["Company"].map(normalize_company)
    old["injury_any_old"] = (pd.to_numeric(old["Number of injuries"], errors="coerce").fillna(0) > 0).astype(int)

    new = feat.copy()
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

    tp = int(((labeled["injury_any_old"] == 1) & (labeled["injury_text_signal"] == 1)).sum())
    fp = int(((labeled["injury_any_old"] == 0) & (labeled["injury_text_signal"] == 1)).sum())
    tn = int(((labeled["injury_any_old"] == 0) & (labeled["injury_text_signal"] == 0)).sum())
    fn = int(((labeled["injury_any_old"] == 1) & (labeled["injury_text_signal"] == 0)).sum())

    return {
        "overlap_rows": int(len(overlap)),
        "labeled_overlap_rows": int(len(labeled)),
        "accuracy": float((labeled["injury_text_signal"] == labeled["injury_any_old"]).mean()) if len(labeled) else float("nan"),
        "positive_precision_proxy": float(tp / max(tp + fp, 1)),
        "positive_recall_proxy": float(tp / max(tp + fn, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#29465b"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    x0, y0 = xy
    ax.text(
        x0 + width / 2,
        y0 + height * 0.68,
        title,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#153243",
    )
    ax.text(
        x0 + width / 2,
        y0 + height * 0.34,
        body,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#20323f",
        linespacing=1.25,
    )


def add_arrow(ax, start, end, color="#61788a"):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.4,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def build_pipeline_figure(stats: dict[str, int]) -> None:
    svg_path = FIGURES_DIR / "Figure1_data_pipeline.svg"
    png_path = FIGURES_DIR / "Figure1_data_pipeline.png"

    palette = {
        "ink": "#111111",
        "body": "#2f2f2f",
        "muted": "#535353",
        "line": "#a8afbb",
        "border": "#bcc7d2",
        "paper": "#ffffff",
        "panel": "#eef2f6",
        "shadow": "#d8dee6",
        "blue": "#6f8eaf",
        "teal": "#1d7074",
        "sand": "#d3a23c",
        "rose": "#be6356",
        "reuse_fill": "#dbe5ef",
        "reuse_chip": "#f6f9fc",
    }
    width = 2048
    height = 856

    def badge(cx: int, cy: int, fill: str, step: str) -> str:
        return f"""
    <g>
      <rect x="{cx-33}" y="{cy-33}" rx="15" ry="15" width="66" height="66" fill="{fill}" filter="url(#badgeShadow)"/>
      <rect x="{cx-30}" y="{cy-30}" rx="13" ry="13" width="60" height="60" fill="{fill}" stroke="#d9e0e7" stroke-width="2"/>
      <text x="{cx}" y="{cy+9}" text-anchor="middle" font-size="32" font-weight="700" fill="#ffffff" font-family="Arial, Helvetica, sans-serif">{step}</text>
    </g>"""

    def card(
        cx: int,
        y: int,
        w: int,
        h: int,
        accent: str,
        step: str,
        title: str,
        value: str,
        subtitle: list[str],
        detail: list[str],
        value_size: int = 58,
        title_size: int = 36,
        subtitle_size: int = 26,
        detail_size: int = 23,
        title_y_offset: int = 94,
        value_y_offset: int = 188,
        subtitle_start_offset: int = 252,
        subtitle_step: int = 34,
        detail_start_offset: int = 292,
        detail_step: int = 28,
    ) -> str:
        x = cx - w // 2
        header_h = 16
        title_y = y + title_y_offset
        value_y = y + value_y_offset
        subtitle_start = y + subtitle_start_offset
        detail_start = y + detail_start_offset
        subtitle_svg = "\n".join(
            f'<text x="{cx}" y="{subtitle_start + i * subtitle_step}" text-anchor="middle" font-size="{subtitle_size}" fill="{palette["body"]}" font-family="Arial, Helvetica, sans-serif">{line}</text>'
            for i, line in enumerate(subtitle)
        )
        detail_svg = "\n".join(
            f'<text x="{cx}" y="{detail_start + i * detail_step}" text-anchor="middle" font-size="{detail_size}" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{line}</text>'
            for i, line in enumerate(detail)
        )
        return f"""
  <g>
    <rect x="{x+10}" y="{y+10}" rx="18" ry="18" width="{w}" height="{h}" fill="{palette["shadow"]}" opacity="0.78"/>
    <rect x="{x}" y="{y}" rx="18" ry="18" width="{w}" height="{h}" fill="{palette["paper"]}" stroke="{palette["border"]}" stroke-width="2.4"/>
    <rect x="{x}" y="{y}" rx="18" ry="18" width="{w}" height="{header_h}" fill="{accent}"/>
    {badge(cx, y, accent, step)}
    <text x="{cx}" y="{title_y}" text-anchor="middle" font-size="{title_size}" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{title}</text>
    <text x="{cx}" y="{value_y}" text-anchor="middle" font-size="{value_size}" font-weight="500" fill="#000000" font-family="Arial, Helvetica, sans-serif">{value}</text>
    {subtitle_svg}
    {detail_svg}
  </g>"""

    def chip(x: int, y: int, w: int, h: int, title: str, subtitle: str) -> str:
        cx = x + w // 2
        return f"""
    <g>
      <rect x="{x}" y="{y}" rx="13" ry="13" width="{w}" height="{h}" fill="{palette["reuse_chip"]}" stroke="{palette["border"]}" stroke-width="1.8"/>
      <text x="{cx}" y="{y+28}" text-anchor="middle" font-size="23" fill="{palette["body"]}" font-family="Arial, Helvetica, sans-serif">{title}</text>
      <text x="{cx}" y="{y+62}" text-anchor="middle" font-size="22" fill="{palette["body"]}" font-family="Arial, Helvetica, sans-serif">{subtitle}</text>
    </g>"""

    reuse_x = 33
    reuse_y = 502
    reuse_w = 974
    reuse_h = 300

    cards_svg = [
        card(
            cx=353,
            y=63,
            w=631,
            h=317,
            accent=palette["blue"],
            step="1",
            title="Public DMV archive",
            value=f"{stats['public_pdfs']}",
            subtitle=["collision-report PDFs"],
            detail=["Public filings, 2019-2026"],
            value_size=82,
            title_size=34,
            subtitle_size=30,
            detail_size=26,
            title_y_offset=93,
            value_y_offset=183,
            subtitle_start_offset=242,
            detail_start_offset=287,
        ),
        card(
            cx=1025,
            y=63,
            w=631,
            h=317,
            accent=palette["blue"],
            step="2",
            title="Form parsing",
            value=f"{stats['widget_rows']:,}",
            subtitle=["widget records"],
            detail=["Audit-preserving extraction"],
            value_size=82,
            title_size=34,
            subtitle_size=30,
            detail_size=26,
            title_y_offset=93,
            value_y_offset=183,
            subtitle_start_offset=242,
            detail_start_offset=287,
        ),
        card(
            cx=1700,
            y=63,
            w=638,
            h=317,
            accent=palette["teal"],
            step="3",
            title="Structured layers",
            value=f"{stats['raw_cols']} / {stats['norm_cols']} / {stats['feat_cols']}",
            subtitle=["raw, normalized, and feature columns"],
            detail=["Traceable machine-readable tables"],
            value_size=78,
            title_size=34,
            subtitle_size=28,
            detail_size=25,
            title_y_offset=93,
            value_y_offset=183,
            subtitle_start_offset=242,
            detail_start_offset=287,
        ),
        card(
            cx=1278,
            y=486,
            w=463,
            h=346,
            accent=palette["sand"],
            step="4",
            title="Eligibility screen",
            value=f"{stats['mode_nonambiguous']} / {stats['injury_labeled']}",
            subtitle=["mode-resolved /", "injury-labeled"],
            detail=["Main-sample screening"],
            value_size=76,
            title_size=34,
            subtitle_size=28,
            detail_size=23,
            title_y_offset=106,
            value_y_offset=206,
            subtitle_start_offset=258,
            subtitle_step=35,
            detail_start_offset=324,
            detail_step=26,
        ),
        card(
            cx=1758,
            y=486,
            w=465,
            h=346,
            accent=palette["rose"],
            step="5",
            title="Inferential sample",
            value=f"{stats['final_sample']}",
            subtitle=["crashes retained"],
            detail=[
                f"Autonomous {stats['mode_autonomous']} • Conventional {stats['mode_conventional']}",
                f"Injury-indicative reports {stats['final_positives']}",
            ],
            value_size=76,
            title_size=34,
            subtitle_size=28,
            detail_size=19,
            title_y_offset=106,
            value_y_offset=206,
            subtitle_start_offset=258,
            subtitle_step=35,
            detail_start_offset=314,
            detail_step=25,
        ),
    ]

    reuse_block = f"""
  <g>
    <rect x="{reuse_x+10}" y="{reuse_y+10}" rx="20" ry="20" width="{reuse_w}" height="{reuse_h}" fill="{palette["shadow"]}" opacity="0.72"/>
    <rect x="{reuse_x}" y="{reuse_y}" rx="20" ry="20" width="{reuse_w}" height="{reuse_h}" fill="{palette["reuse_fill"]}" stroke="{palette["border"]}" stroke-width="2"/>
    <rect x="{reuse_x}" y="{reuse_y}" rx="20" ry="20" width="{reuse_w}" height="16" fill="#91a8bc"/>
    <text x="{reuse_x + reuse_w/2}" y="{reuse_y + 80}" text-anchor="middle" font-size="34" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">Intermediate data products retained for reuse</text>
    {chip(reuse_x + 65, reuse_y + 126, 279, 84, "Raw widget table", "traceability")}
    {chip(reuse_x + 354, reuse_y + 126, 279, 84, "Long audit table", "quality control")}
    {chip(reuse_x + 643, reuse_y + 126, 279, 84, "Normalized crash table", "future hypotheses")}
    <text x="{reuse_x + reuse_w/2}" y="{reuse_y + 254}" text-anchor="middle" font-size="24" fill="{palette["body"]}" font-family="Arial, Helvetica, sans-serif">The full parsing stack is preserved so later hypotheses can</text>
    <text x="{reuse_x + reuse_w/2}" y="{reuse_y + 286}" text-anchor="middle" font-size="24" fill="{palette["body"]}" font-family="Arial, Helvetica, sans-serif">be tested without re-extracting the PDFs.</text>
  </g>"""

    arrows = f"""
  <g stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M668 222 L703 222" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M703 222 l-16 -14 M703 222 l-16 14" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M1341 222 L1377 222" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M1377 222 l-16 -14 M1377 222 l-16 14" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M1710 381 C1710 438, 1710 438, 1618 438 L525 438 C525 438, 525 462, 525 462" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M525 462 l-14 -14 M525 462 l14 -14" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M1007 652 L1046 652" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M1046 652 l-16 -14 M1046 652 l-16 14" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M1510 652 L1528 652" stroke="{palette["line"]}" stroke-width="3.2"/>
    <path d="M1528 652 l-16 -14 M1528 652 l-16 14" stroke="{palette["line"]}" stroke-width="3.2"/>
  </g>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <filter id="badgeShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#9ca8b5" flood-opacity="0.55"/>
    </filter>
  </defs>
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <rect x="1" y="1" rx="42" ry="42" width="{width-2}" height="{height-2}" fill="{palette["panel"]}" stroke="#e1e7ee" stroke-width="2"/>
  {cards_svg[0]}
  {cards_svg[1]}
  {cards_svg[2]}
  {reuse_block}
  {cards_svg[3]}
  {cards_svg[4]}
  {arrows}
</svg>
"""

    svg_path.write_text(svg, encoding="utf-8")
    subprocess.run(["sips", "-s", "format", "png", str(svg_path), "--out", str(png_path)], check=True, capture_output=True, text=True)


def build_archive_profile_figure(feat: pd.DataFrame, stats: dict[str, int]) -> None:
    fig = plt.figure(figsize=(13.5, 7.8), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.0], hspace=0.34, wspace=0.24)

    year_ax = fig.add_subplot(gs[0, :])
    mode_ax = fig.add_subplot(gs[1, 0])
    manu_ax = fig.add_subplot(gs[1, 1])

    year_counts = (
        feat.dropna(subset=["accident_year"])
        .groupby("accident_year")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("accident_year")
    )
    year_counts["accident_year"] = year_counts["accident_year"].astype(int)
    year_colors = ["#7aa6c2" if y < 2024 else "#d08159" for y in year_counts["accident_year"]]
    year_pos = list(range(len(year_counts)))
    year_labels = [str(y) for y in year_counts["accident_year"]]
    year_ax.bar(year_pos, year_counts["count"], color=year_colors, width=0.72)
    year_ax.set_xticks(year_pos)
    year_ax.set_xticklabels(year_labels)
    for x, y in zip(year_pos, year_counts["count"]):
        year_ax.text(x, y + 3, f"{int(y)}", ha="center", va="bottom", fontsize=9.5, color="#314555")
    year_ax.text(0.0, 1.04, "A", transform=year_ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")
    year_ax.set_ylabel("Number of reports")
    year_ax.grid(axis="y", linestyle="--", alpha=0.25)

    mode_counts = (
        feat["mode_resolved"]
        .fillna("missing")
        .replace({"ambiguous": "ambiguous", "missing": "missing"})
        .value_counts()
        .reindex(["autonomous", "conventional", "ambiguous", "missing"])
        .fillna(0)
        .astype(int)
    )
    mode_colors = ["#4878a8", "#d08159", "#b7aa9a", "#d8d8d8"]
    mode_ax.bar(range(len(mode_counts)), mode_counts.values, color=mode_colors, width=0.66)
    mode_ax.set_xticks(range(len(mode_counts)))
    mode_ax.set_xticklabels(["Autonomous", "Conventional", "Ambiguous", "Missing"], rotation=12)
    mode_ax.text(0.0, 1.04, "B", transform=mode_ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")
    mode_ax.set_ylabel("Number of reports")
    mode_ax.grid(axis="y", linestyle="--", alpha=0.25)
    for idx, val in enumerate(mode_counts.values):
        mode_ax.text(idx, val + 4, str(val), ha="center", va="bottom", fontsize=9.5, color="#314555")

    top_full = feat["manufacturer_std"].fillna("Unknown").value_counts().head(6)
    final = feat[feat["mode_resolved"].isin(["autonomous", "conventional"])].copy()
    final = final[final["injury_text_signal"].notna()].copy()
    top_names = top_full.index.tolist()
    final_counts = final["manufacturer_std"].fillna("Unknown").value_counts().reindex(top_names).fillna(0).astype(int)
    y = list(range(len(top_names)))
    manu_ax.barh([i + 0.18 for i in y], top_full.values[::-1], height=0.34, color="#90b8cf", label="Parsed archive")
    manu_ax.barh([i - 0.18 for i in y], final_counts.values[::-1], height=0.34, color="#cf7f62", label="Main sample")
    manu_ax.set_yticks(y)
    manu_ax.set_yticklabels(top_names[::-1])
    manu_ax.text(0.0, 1.04, "C", transform=manu_ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")
    manu_ax.set_xlabel("Number of reports")
    manu_ax.legend(frameon=False, loc="lower right")
    manu_ax.grid(axis="x", linestyle="--", alpha=0.25)

    fig.savefig(FIGURES_DIR / "Figure2_archive_profile.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_data_quality_figure(feat: pd.DataFrame, validation: dict[str, object]) -> None:
    final = feat[feat["mode_resolved"].isin(["autonomous", "conventional"])].copy()
    final = final[final["injury_text_signal"].notna()].copy()

    coverage_rows = [
        ("Manufacturer", "manufacturer_std"),
        ("Crash year", "accident_year"),
        ("Crash time", "accident_time"),
        ("Resolved mode", "mode_resolved"),
        ("Injury label", "injury_text_signal"),
        ("Rear-end indicator", "collision_type__rear_end"),
        ("Stopped cue", "narrative_mentions_stopped"),
        ("Intersection cue", "location_is_intersection_like"),
        ("Night indicator", "is_night"),
        ("VRU indicator", "vru_any"),
        ("Damage severity", "vehicle1_damage_ordinal"),
    ]
    cover = pd.DataFrame(
        {
            "label": [label for label, _ in coverage_rows],
            "archive": [float(feat[column].notna().mean()) for _, column in coverage_rows],
            "sample": [float(final[column].notna().mean()) for _, column in coverage_rows],
        }
    )
    cover = cover.iloc[::-1].reset_index(drop=True)

    fig = plt.figure(figsize=(13.4, 6.7), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.62, 0.95], wspace=0.18)
    cov_ax = fig.add_subplot(gs[0, 0])
    cm_ax = fig.add_subplot(gs[0, 1])

    y = range(len(cover))
    cov_ax.barh([i + 0.18 for i in y], cover["archive"], height=0.32, color="#9ebed2", label="Parsed archive")
    cov_ax.barh([i - 0.18 for i in y], cover["sample"], height=0.32, color="#cf7f62", label="Main sample")
    cov_ax.set_yticks(list(y))
    cov_ax.set_yticklabels(cover["label"])
    cov_ax.set_xlim(0, 1.02)
    cov_ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    cov_ax.grid(axis="x", linestyle="--", alpha=0.25)
    cov_ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.54, 1.12), ncol=2)
    cov_ax.text(0.0, 1.03, "A", transform=cov_ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")
    for i, (_, row) in enumerate(cover.iterrows()):
        cov_ax.text(min(row["archive"] + 0.012, 0.995), i + 0.18, f"{row['archive']:.0%}", va="center", fontsize=9, color="#435c6d")
        cov_ax.text(min(row["sample"] + 0.012, 0.995), i - 0.18, f"{row['sample']:.0%}", va="center", fontsize=9, color="#7a3d2c")

    cm = [[validation["tn"], validation["fp"]], [validation["fn"], validation["tp"]]]
    cm_ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(validation["tn"], validation["tp"]))
    cm_ax.set_xticks([0, 1])
    cm_ax.set_yticks([0, 1])
    cm_ax.set_xticklabels(["No injury", "Injury"])
    cm_ax.set_yticklabels(["No injury", "Injury"])
    cm_ax.set_xlabel("Narrative weak label")
    cm_ax.set_ylabel("Legacy structured label", labelpad=2)
    cm_ax.text(0.0, 1.03, "B", transform=cm_ax.transAxes, fontsize=13, fontweight="bold", color="#1f3442")
    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            text_color = "white" if value > 0.45 * max(validation["tn"], validation["tp"]) else "#183147"
            cm_ax.text(j, i, str(value), ha="center", va="center", fontsize=13, fontweight="bold", color=text_color)
    for spine in cm_ax.spines.values():
        spine.set_visible(False)

    fig.savefig(FIGURES_DIR / "Figure3_data_quality.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_sample_construction_table(stats: dict[str, int]) -> None:
    rows = [
        ("Public PDF collision reports parsed", stats["public_pdfs"], 1.000),
        ("Reports with non-ambiguous mode (autonomous or conventional)", stats["mode_nonambiguous"], stats["mode_nonambiguous"] / stats["public_pdfs"]),
        ("Reports with non-missing narrative injury label", stats["injury_labeled"], stats["injury_labeled"] / stats["public_pdfs"]),
        ("Final inferential sample (mode + injury label)", stats["final_sample"], stats["final_sample"] / stats["public_pdfs"]),
        ("Injury cases in final inferential sample", stats["final_positives"], stats["final_positives"] / stats["public_pdfs"]),
    ]

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Sample construction for the PDF-based analysis.}",
        "\\label{tab:sample_construction}",
        "\\small",
        "\\begin{tabular}{p{7.3cm}rr}",
        "\\toprule",
        "Stage & Count & Share of parsed archive \\\\",
        "\\midrule",
    ]
    for stage, count, share in rows:
        lines.append(f"{stage} & {count} & {share:.3f} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    (TABLES_DIR / "Table1_sample_construction.tex").write_text("\n".join(lines) + "\n")


def write_variable_table(feat: pd.DataFrame) -> None:
    final = feat[feat["mode_resolved"].isin(["autonomous", "conventional"])].copy()
    final = final[final["injury_text_signal"].notna()].copy()

    rows = [
        ("Injury indicator", "Narrative-derived weak label for injury versus explicit no-injury language.", "injury_text_signal"),
        ("Rear-end crash", "Harmonized indicator from Page-3 collision-type coding.", "collision_type__rear_end"),
        ("Autonomous mode", "Resolved autonomous versus conventional mode; ambiguous records excluded.", "mode_resolved"),
        ("Stopped-traffic cue", "Narrative cue for stopped, stopping, or queue-like traffic.", "narrative_mentions_stopped"),
        ("Intersection-like location", "Indicator for intersection or intersection-related location wording.", "location_is_intersection_like"),
        ("Night indicator", "Indicator derived from parsed crash time.", "is_night"),
        ("VRU involvement", "Indicator for pedestrian, bicyclist, scooter, or similar vulnerable users.", "vru_any"),
        ("Damage severity", "Ordinal damage scale for the AV or reporting vehicle.", "vehicle1_damage_ordinal"),
    ]

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Paper-facing variable definitions and availability.}",
        "\\label{tab:key_variables}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{p{2.35cm}p{5.2cm}rr}",
        "\\toprule",
        "Variable & Operational definition & Archive avail. & Main-sample avail. \\\\",
        "\\midrule",
    ]
    for label, definition, column in rows:
        archive_avail = float(feat[column].notna().mean())
        sample_avail = float(final[column].notna().mean())
        lines.append(f"{label} & {definition} & {archive_avail:.3f} & {sample_avail:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table2_key_variables.tex").write_text("\n".join(lines) + "\n")


def write_validation_table(validation: dict[str, object]) -> None:
    rows = [
        ("One-to-one archive overlap", int(validation["overlap_rows"])),
        ("Overlap rows with non-missing weak label", int(validation["labeled_overlap_rows"])),
        ("Agreement with legacy injury label", f"{validation['accuracy']:.3f}"),
        ("Positive precision proxy", f"{validation['positive_precision_proxy']:.3f}"),
        ("Positive recall proxy", f"{validation['positive_recall_proxy']:.3f}"),
        ("True negatives / false positives", f"{validation['tn']} / {validation['fp']}"),
        ("False negatives / true positives", f"{validation['fn']} / {validation['tp']}"),
    ]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Weak-label validation against the legacy structured California DMV crash table.}",
        "\\label{tab:weak_label_validation}",
        "\\small",
        "\\begin{tabular}{p{7.4cm}r}",
        "\\toprule",
        "Validation quantity & Value \\\\",
        "\\midrule",
    ]
    for item, value in rows:
        lines.append(f"{item} & {value} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES_DIR / "Table3_weak_label_validation.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    feat, stats = load_data()
    validation = build_validation_summary(feat)
    build_pipeline_figure(stats)
    build_archive_profile_figure(feat, stats)
    build_data_quality_figure(feat, validation)
    write_sample_construction_table(stats)
    write_variable_table(feat)
    write_validation_table(validation)


if __name__ == "__main__":
    main()
