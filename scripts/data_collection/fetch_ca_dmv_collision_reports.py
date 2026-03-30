#!/usr/bin/env python3
"""Download public California DMV autonomous vehicle collision report PDFs.

This script mirrors the publicly linked PDFs from the California DMV collision
reports page into a local directory, grouped by report year, and writes a CSV
manifest for downstream parsing.

Important limitation:
    The California DMV page states that collision reports dated before
    2019-01-01 are archived and must be requested by email. This downloader can
    only fetch the report PDFs that are publicly linked on the page.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
LIST_PAGE_URL = (
    "https://www.dmv.ca.gov/portal/vehicle-industry-services/"
    "autonomous-vehicles/autonomous-vehicle-collision-reports/"
)
FORM_SLUG = "report-of-traffic-accident-involving-an-autonomous-vehicle-ol-316-pdf"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/133.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.dmv.ca.gov/",
}
READ_CHUNK_SIZE = 1024 * 1024
MONTH_TOKENS = {
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
}


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


class CollisionReportLinkParser(HTMLParser):
    """Extract report links from the DMV collision reports page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._current_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href")
        if not href or "/portal/file/" not in href:
            return

        self._current_href = href
        self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is None:
            return
        self._current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current_href is None:
            return

        text = normalize_whitespace(" ".join(self._current_text_parts))
        self.links.append((self._current_href, text))
        self._current_href = None
        self._current_text_parts = []


def fetch_bytes(url: str, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    request = Request(url, headers=DEFAULT_HEADERS)

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, socket.timeout, http.client.HTTPException) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"Retrying after error on {url}: {exc} (attempt {attempt}/{retries})",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_text(url: str, timeout: int, retries: int) -> str:
    return fetch_bytes(url, timeout=timeout, retries=retries).decode("utf-8", errors="replace")


def parse_report_date(text: str) -> Optional[datetime]:
    cleaned = text.replace("(PDF)", "").replace("Narrative", "")
    cleaned = cleaned.replace("(A)", "").replace("(B)", "")
    cleaned = normalize_whitespace(cleaned)

    tokens = cleaned.split()
    for start in range(len(tokens)):
        for width in (3, 4):
            chunk = " ".join(tokens[start : start + width]).replace(",", "").replace(".", "")
            for fmt in ("%B %d %Y", "%b %d %Y"):
                try:
                    return datetime.strptime(chunk, fmt)
                except ValueError:
                    continue
    return None


def sanitize_slug(url: str) -> str:
    slug = Path(urlparse(url).path.rstrip("/")).name
    return slug or "unknown-report"


@dataclass(frozen=True)
class CollisionReport:
    title: str
    page_url: str

    @property
    def slug(self) -> str:
        return sanitize_slug(self.page_url)

    @property
    def is_form(self) -> bool:
        return self.slug == FORM_SLUG

    @property
    def report_date(self) -> Optional[datetime]:
        return parse_report_date(self.title)

    @property
    def year_folder(self) -> str:
        if self.report_date is not None:
            return str(self.report_date.year)
        for piece in self.title.split():
            if piece.isdigit() and len(piece) == 4:
                return piece
        return "unknown_year"

    @property
    def manufacturer(self) -> str:
        tokens = self.title.split()
        for index, token in enumerate(tokens):
            cleaned = token.strip(",.()").casefold()
            if cleaned in MONTH_TOKENS:
                return " ".join(tokens[:index]).strip()
        return ""

    @property
    def filename(self) -> str:
        slug = self.slug
        return slug if slug.lower().endswith(".pdf") else f"{slug}.pdf"


def list_reports(timeout: int, retries: int, include_form: bool) -> list[CollisionReport]:
    html_text = fetch_text(LIST_PAGE_URL, timeout=timeout, retries=retries)
    parser = CollisionReportLinkParser()
    parser.feed(html_text)

    reports_by_url: dict[str, CollisionReport] = {}
    for raw_href, raw_text in parser.links:
        report = CollisionReport(
            title=normalize_whitespace(raw_text),
            page_url=urljoin(LIST_PAGE_URL, raw_href),
        )
        if report.is_form and not include_form:
            continue
        reports_by_url.setdefault(report.page_url, report)

    reports = list(reports_by_url.values())
    reports.sort(
        key=lambda report: (
            report.year_folder,
            report.report_date.isoformat() if report.report_date else "",
            report.slug,
        )
    )
    return reports


def stream_download(url: str, destination: Path, timeout: int, retries: int) -> None:
    last_error: Exception | None = None
    request = Request(url, headers=DEFAULT_HEADERS)

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
                while True:
                    chunk = response.read(READ_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
            return
        except (HTTPError, URLError, TimeoutError, socket.timeout, http.client.HTTPException) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt == retries:
                break
            sleep_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"Retrying after error on {url}: {exc} (attempt {attempt}/{retries})",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def write_manifest(manifest_path: Path, reports: list[CollisionReport], output_root: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "manufacturer",
                "report_date",
                "year_folder",
                "page_url",
                "slug",
                "local_path",
            ],
        )
        writer.writeheader()
        for report in reports:
            local_path = output_root / report.year_folder / report.filename
            writer.writerow(
                {
                    "title": report.title,
                    "manufacturer": report.manufacturer,
                    "report_date": report.report_date.date().isoformat()
                    if report.report_date
                    else "",
                    "year_folder": report.year_folder,
                    "page_url": report.page_url,
                    "slug": report.slug,
                    "local_path": str(local_path),
                }
            )


def download_reports(
    reports: list[CollisionReport],
    output_root: Path,
    timeout: int,
    retries: int,
    overwrite: bool,
    sleep_seconds: float,
) -> tuple[int, int]:
    downloaded = 0
    skipped = 0

    for index, report in enumerate(reports, start=1):
        destination = output_root / report.year_folder / report.filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not overwrite:
            skipped += 1
            print(f"[{index}/{len(reports)}] skip {destination}")
            continue

        temp_path = destination.with_name(destination.name + ".part")
        temp_path.unlink(missing_ok=True)

        print(f"[{index}/{len(reports)}] download {report.page_url}")
        try:
            stream_download(report.page_url, temp_path, timeout=timeout, retries=retries)
            temp_path.replace(destination)
            downloaded += 1
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return downloaded, skipped


def filter_reports(
    reports: list[CollisionReport],
    years: Optional[set[int]],
    match: Optional[str],
    limit: Optional[int],
) -> list[CollisionReport]:
    filtered = reports

    if years:
        filtered = [report for report in filtered if report.year_folder.isdigit() and int(report.year_folder) in years]

    if match:
        needle = match.casefold()
        filtered = [
            report
            for report in filtered
            if needle in report.title.casefold() or needle in report.slug.casefold()
        ]

    if limit is not None:
        filtered = filtered[:limit]

    return filtered


def print_summary(reports: list[CollisionReport]) -> None:
    counts: dict[str, int] = {}
    for report in reports:
        counts[report.year_folder] = counts.get(report.year_folder, 0) + 1

    print(f"Discovered {len(reports)} downloadable DMV collision PDFs.")
    for year_folder in sorted(counts):
        print(f"  {year_folder}: {counts[year_folder]}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the publicly linked collision report PDFs from the California DMV "
            "Autonomous Vehicle Collision Reports page and write a CSV manifest."
        )
    )
    parser.add_argument(
        "--output",
        default=str(PACKAGE_ROOT / "reports"),
        help="Directory where year-based PDF folders will be created.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path for the CSV manifest. Defaults to <output>/manifest.csv.",
    )
    parser.add_argument(
        "--year",
        action="append",
        type=int,
        help="Restrict downloads to a specific report year. Repeatable.",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Only keep reports whose title or slug contains this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N filtered reports. Useful for testing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Network timeout in seconds for each request. Default: 300.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for each HTTP request. Default: 3.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay in seconds between downloads. Default: 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only crawl the page, print a summary, and write the manifest.",
    )
    parser.add_argument(
        "--include-form",
        action="store_true",
        help="Also include the blank OL 316 form PDF that appears on the page.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    output_root = Path(args.output).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_root / "manifest.csv"
    )

    reports = list_reports(timeout=args.timeout, retries=args.retries, include_form=args.include_form)
    reports = filter_reports(
        reports,
        years=set(args.year) if args.year else None,
        match=args.match,
        limit=args.limit,
    )

    print_summary(reports)
    write_manifest(manifest_path, reports, output_root)
    print(f"Manifest written to {manifest_path}")
    print(
        "Note: pre-2019 reports are archived by DMV and are not publicly linked as direct PDFs.",
        file=sys.stderr,
    )

    if args.list_only:
        return 0

    downloaded, skipped = download_reports(
        reports=reports,
        output_root=output_root,
        timeout=args.timeout,
        retries=args.retries,
        overwrite=args.overwrite,
        sleep_seconds=args.sleep,
    )
    print(f"Completed: downloaded={downloaded}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
