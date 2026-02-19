---
phase: 02-data-acquisition-and-audit
verified: 2026-02-19T21:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 2: Data Acquisition and Audit Verification Report

**Phase Goal:** The raw BTXRD dataset is downloaded, organized, profiled, and documented -- all data quality issues are surfaced before any modeling begins
**Verified:** 2026-02-19T21:00:00Z
**Status:** PASSED
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | `python scripts/download.py --config configs/default.yaml` downloads BTXRD.zip from figshare, verifies MD5, extracts, and organizes files into data_raw/ | VERIFIED | Script exists (351 lines), full download pipeline with streaming HTTP, incremental MD5, zip extraction, xlsx-to-csv conversion, and wrapper directory flattening. Makefile `download` target calls it. |
| 2  | data_raw/images/ contains JPEG radiograph files | VERIFIED | Directory exists with exactly 3,746 files (IMG000001.jpeg through IMG003746.jpeg). Confirmed via `ls | wc -l`. |
| 3  | data_raw/dataset.csv exists and is a valid CSV with rows and 37 columns | VERIFIED | File has 3,747 lines (header + 3,746 rows), 37 columns confirmed by python csv reader. All 37 column names match spec. |
| 4  | data_raw/Annotations/ contains LabelMe JSON files for tumor images | VERIFIED | Directory exists with exactly 1,867 JSON files matching the 1,867 tumor images in dataset.csv. 100% annotation coverage confirmed. |
| 5  | Re-running the download script when data already exists skips the download with a message | VERIFIED | `check_existing_data()` at line 57 returns True when images/, Annotations/, and dataset.csv all present. `main()` at line 304 logs "Data already exists at %s. Skipping download." and returns early. |
| 6  | MD5 verification catches corrupt downloads and raises a clear error | VERIFIED | `download_with_verification()` raises `ValueError("MD5 mismatch: expected {expected_md5}, got {actual_md5}. Corrupt download has been deleted.")` at line 143-146. Corrupt file is deleted before raising. |
| 7  | `python scripts/audit.py --config configs/default.yaml` generates docs/data_audit_report.md with embedded figures | VERIFIED | Script exists (706 lines), generates 7-section report. docs/data_audit_report.md exists (191 lines), auto-generated on 2026-02-19 20:30:38. All 5 figure PNG files present in docs/figures/. |
| 8  | Audit report confirms class distribution 1,879 Normal / 1,525 Benign / 342 Malignant; includes image dimension histogram, per-center breakdown, missing values, annotation coverage, and duplicate detection | VERIFIED | Report contains all 6 required sections. Numbers confirmed: 1,879/1,525/342 exact match. 5 embedded figures: class_distribution.png, dimension_histogram.png, center_breakdown.png, annotation_coverage.png, duplicate_detection.png. |
| 9  | Audit report explicitly documents leakage risk from same-lesion multi-angle images (no patient_id) and notes proxy grouping columns | VERIFIED | Section 7 "Data Leakage Risk Assessment" at line 157: "no `patient_id` column exists", risk statement, shooting angle distribution (frontal 2,181/lateral 1,269/oblique 296), proxy grouping table (295 groups via center+age+gender, 941 groups with site added), and mitigation recommendation. |
| 10 | docs/dataset_spec.md documents all 37 columns using actual CSV header names, label derivation logic (malignant=1->Malignant, benign=1->Benign, tumor=0->Normal), data provenance per center, and CC BY-NC-ND 4.0 license | VERIFIED | File exists (281 lines). All 37 columns documented with actual header names, types, descriptions, and value counts. Label derivation table at line 133-141 matches exact priority logic. Data Provenance Per Center table at line 168. CC BY-NC-ND 4.0 documented at line 12 with discrepancy note re figshare CC BY metadata. |

**Score:** 10/10 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/download.py` | Download pipeline with MD5, skip-if-exists, clear errors | VERIFIED | 351 lines, exports no component (script entry point). Full implementation: `check_existing_data()`, `download_with_verification()`, `extract_and_organize()`, `main()`. No stubs or placeholders. |
| `scripts/audit.py` | Audit pipeline generating report with all 7 sections | VERIFIED | 706 lines. Seven section functions + report assembly. All sections return (figure_path, markdown) tuples. Wired to docs_dir via config. |
| `data_raw/images/` | 3,746 JPEG radiograph files | VERIFIED | EXISTS. 3,746 files confirmed. |
| `data_raw/Annotations/` | 1,867 LabelMe JSON annotation files | VERIFIED | EXISTS. 1,867 files confirmed. 100% tumor image coverage. |
| `data_raw/dataset.csv` | 3,746 rows, 37 columns, valid CSV | VERIFIED | EXISTS. 3,747 lines (header + 3,746 rows). 37 columns verified. Normal: 1,879, Benign: 1,525, Malignant: 342. |
| `docs/data_audit_report.md` | Auto-generated report with 7 sections and 5 embedded figures | VERIFIED | EXISTS. 191 lines. Auto-generated on 2026-02-19. 7 sections, 5 figure references (all PNGs confirmed present). |
| `docs/dataset_spec.md` | 37-column spec, label derivation, provenance, license | VERIFIED | EXISTS. 281 lines. Comprehensive -- all 37 columns by actual CSV name, label derivation table and Python code, per-center provenance, CC BY-NC-ND 4.0 with discrepancy note. |
| `docs/figures/class_distribution.png` | Bar chart of class counts | VERIFIED | EXISTS. |
| `docs/figures/dimension_histogram.png` | Width and height distributions | VERIFIED | EXISTS. |
| `docs/figures/center_breakdown.png` | Stacked bar by center | VERIFIED | EXISTS. |
| `docs/figures/annotation_coverage.png` | Annotation coverage chart | VERIFIED | EXISTS. |
| `docs/figures/duplicate_detection.png` | Pairwise hash distance histogram | VERIFIED | EXISTS. |
| `configs/default.yaml` | figshare_url, expected_md5, raw_dir configured | VERIFIED | figshare_url: https://ndownloader.figshare.com/files/50653575, expected_md5: 59132a5036d030580ccade8add8f13df, raw_dir: data_raw. All present. |
| `Makefile` | `download` and `audit` targets wired to scripts | VERIFIED | `make download` calls `python scripts/download.py --config configs/default.yaml`. `make audit` calls `python scripts/audit.py --config configs/default.yaml`. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `make download` | `scripts/download.py` | Makefile target | WIRED | Line 9-10: `download: ## ...` -> `python scripts/download.py --config configs/default.yaml` |
| `make audit` | `scripts/audit.py` | Makefile target | WIRED | Line 12-13: `audit: ## ...` -> `python scripts/audit.py --config configs/default.yaml` |
| `download.py` | figshare URL | `configs/default.yaml` -> `cfg["data"]["figshare_url"]` | WIRED | Line 298: `figshare_url = cfg["data"]["figshare_url"]`. Config has correct URL. |
| `download.py` | MD5 verification | `cfg["data"]["expected_md5"]` | WIRED | Line 299: `expected_md5 = cfg["data"]["expected_md5"]`. Raises ValueError on mismatch. |
| `download.py` | `data_raw/` | `cfg["data"]["raw_dir"]` | WIRED | Line 297: `raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]`. raw_dir=data_raw in config. |
| `audit.py` | `data_raw/dataset.csv` | `cfg["data"]["raw_dir"]` | WIRED | Line 620-622: resolves raw_dir, images_dir, csv_path. Validates all exist before proceeding. |
| `audit.py` | `docs/data_audit_report.md` | `cfg["paths"]["docs_dir"]` | WIRED | Line 623: `docs_dir = PROJECT_ROOT / cfg["paths"]["docs_dir"]`. Line 696: `report_path = docs_dir / "data_audit_report.md"`. |
| `audit.py` | `docs/figures/*.png` | `figures_dir = docs_dir / "figures"` | WIRED | Line 624: `figures_dir = docs_dir / "figures"`. `save_figure()` creates dir and saves all 5 PNGs. |
| `audit_leakage_risk()` | Leakage section in report | `sections.append()` at line 693 | WIRED | Returns markdown string. Appended to sections with title "Data Leakage Risk Assessment". Assembled into report by `write_audit_report()`. |

---

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|---------|
| DATA-01: Download script fetches BTXRD dataset from figshare and organizes into data_raw/ | SATISFIED | scripts/download.py fetches from figshare URL, verifies MD5, extracts to data_raw/ with images/, Annotations/, dataset.csv. Data confirmed present. |
| DATA-02: Data audit report covering class distribution, image dimension histogram, missing values, annotation coverage, duplicate detection, per-center breakdown | SATISFIED | All 6 sections present in docs/data_audit_report.md with real data (not stubs). Exact numbers confirmed from live CSV. |
| DATA-03: Dataset specification document describing all 37 columns, label derivation logic, and data provenance | SATISFIED | docs/dataset_spec.md covers all 37 columns (using actual CSV header names), label derivation priority table with Python code, per-center provenance table. |
| DOCS-01: dataset_spec.md with column definitions, label schema, data provenance, license | SATISFIED | All 4 elements present: 37-column definitions, label derivation schema, 3-center provenance, CC BY-NC-ND 4.0 with figshare discrepancy note. |
| DOCS-02: data_audit_report.md auto-generated from audit script with embedded figures | SATISFIED | Report has auto-generation timestamp header. All 5 figures embedded with `![Title](figures/name.png)` syntax. Figures confirmed present on disk. |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No stubs, TODOs, placeholders, or empty handlers found in download.py or audit.py. Both scripts are fully implemented. |

---

## Human Verification Required

### 1. Actual Download Execution

**Test:** Run `make download` on a machine where data_raw/ does not exist (or temporarily rename it), and observe the download progress.
**Expected:** Downloads BTXRD.zip (~801MB) from figshare with tqdm progress bar, verifies MD5, extracts and organizes to data_raw/.
**Why human:** The data is already present. Programmatic verification confirms the script logic is correct and the data is organized, but a fresh download cannot be triggered without deleting existing data.

### 2. Audit Report Figure Quality

**Test:** Open docs/figures/*.png and visually inspect each chart.
**Expected:** Class distribution bar chart shows clearly labeled bars for Normal (1,879), Benign (1,525), Malignant (342). Histogram shows bimodal or wide distribution for both width and height. Stacked bar shows center breakdown. Annotation coverage shows 100% covered bar. Duplicate detection shows histogram with threshold line.
**Why human:** Programmatic checks confirm files exist and scripts generate them, but visual quality (axis labels, readability, correct rendering) requires human inspection.

---

## Summary

Phase 2 goal is fully achieved. Every must-have truth is verified at all three levels (existence, substantive implementation, correct wiring).

**Download pipeline (02-01):** scripts/download.py is a complete 351-line implementation with streaming HTTP download, incremental MD5 verification, ZIP extraction with wrapper directory flattening, xlsx-to-csv conversion (the actual dataset ships as Excel), nested Annotations/ cleanup, and idempotent re-run protection. The data_raw/ directory is fully populated: 3,746 JPEG images, 1,867 LabelMe JSON annotations, and a 37-column dataset.csv confirmed to contain exactly 1,879 Normal / 1,525 Benign / 342 Malignant rows.

**Audit pipeline (02-02):** scripts/audit.py is a complete 706-line implementation generating a 7-section markdown report. The report (docs/data_audit_report.md) is auto-generated and confirmed to contain all required sections: class distribution with exact counts, image dimension histogram (153-3594px width, 311-4881px height), per-center breakdown, zero missing values across 142,348 cells, 100% annotation coverage, duplicate detection (21 exact + 20 near-duplicate pairs), and a detailed leakage risk assessment. All 5 PNG figures are present in docs/figures/.

**Dataset specification (02-02):** docs/dataset_spec.md documents all 37 columns using actual CSV header names (including space-delimited names like "hip bone", "simple bone cyst", and hyphenated names like "ankle-joint"), the label derivation priority logic (malignant=1 -> Malignant, benign=1 -> Benign, else Normal), per-center data provenance (Center 1: Guangxi Medical University, Center 2: Radiopaedia, Center 3: MedPix), and the CC BY-NC-ND 4.0 license with a note on the figshare CC BY discrepancy.

**Leakage risk:** The audit report section "Data Leakage Risk Assessment" explicitly states "no `patient_id` column exists", explains the risk from multi-angle images, documents proxy grouping options (center+age+gender: 295 groups; with site: 941 groups), and recommends the center-holdout split as the primary mitigation.

No gaps, stubs, or blockers found. Phase 3 can proceed.

---

_Verified: 2026-02-19T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
