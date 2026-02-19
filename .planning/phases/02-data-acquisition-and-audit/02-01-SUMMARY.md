---
phase: 02-data-acquisition-and-audit
plan: 01
subsystem: data
tags: [figshare, download, md5, zip-extraction, xlsx-to-csv, btxrd, requests, tqdm]

# Dependency graph
requires:
  - phase: 01-02
    provides: config system, placeholder scripts, Makefile targets
provides:
  - "Download script (scripts/download.py) fetching BTXRD from figshare with MD5 verification"
  - "Corrected configs/default.yaml with figshare file URL, MD5 hash, raw_dir=data_raw"
  - "data_raw/ populated: 3746 JPEG images, 1867 JSON annotations, dataset.csv (37 cols)"
  - "requirements.txt updated with requests, imagehash, openpyxl"
affects: [02-02, 03-01, 03-02, 04-01, 05-01, 06-01]

# Tech tracking
tech-stack:
  added: [requests, imagehash, openpyxl]
  patterns: [streaming-download-with-tqdm, incremental-md5, zip-wrapper-flattening, xlsx-to-csv-conversion, idempotent-skip]

key-files:
  created: [data_raw/dataset.csv, data_raw/images/, data_raw/Annotations/]
  modified: [configs/default.yaml, requirements.txt, scripts/download.py]

key-decisions:
  - "Convert dataset.xlsx to dataset.csv during extraction (ZIP ships Excel, not CSV)"
  - "Remove nested Annotations/Annotations/ directory (ZIP artifact)"
  - "Add openpyxl to requirements.txt for xlsx reading"
  - "Skip re-download when ZIP already exists and MD5 matches"
  - "Use assurexray.download logger name for parent logger inheritance"

patterns-established:
  - "Streaming download with tqdm progress bar and incremental MD5 verification"
  - "ZIP extraction with top-level wrapper directory flattening"
  - "Idempotent data download (check_existing_data before download)"
  - "JPEG glob: *.[jJ][pP][gG] + *.[jJ][pP][eE][gG] for case-insensitive matching"

# Metrics
duration: 11min
completed: 2026-02-19
---

# Phase 2 Plan 1: Download BTXRD Dataset Summary

**Streaming download of 801MB BTXRD.zip from figshare with MD5 verification, xlsx-to-csv conversion, and ZIP wrapper flattening into data_raw/ (3746 images, 1867 annotations, 37-column CSV)**

## Performance

- **Duration:** 11 min
- **Started:** 2026-02-19T20:07:59Z
- **Completed:** 2026-02-19T20:19:13Z
- **Tasks:** 2
- **Files modified:** 3 (configs/default.yaml, requirements.txt, scripts/download.py)

## Accomplishments

- Complete download pipeline: streaming HTTP with tqdm progress, incremental MD5, ZIP extraction with wrapper flattening, xlsx-to-csv conversion
- data_raw/ contains 3,746 JPEG radiographs, 1,867 LabelMe JSON annotations, and dataset.csv with 3,746 rows x 37 columns
- Class distribution confirmed: 1,879 Normal, 1,525 Benign, 342 Malignant (matches paper)
- Idempotent: re-running skips with "Data already exists" message
- Actual column names confirmed (critical for Plan 02-02): uses spaces for multi-word names (e.g., `hip bone`, `simple bone cyst`), hyphens for joints (e.g., `ankle-joint`), no underscores

## Task Commits

Each task was committed atomically:

1. **Task 1: Update config/deps and implement download.py** - `307f7e2` (feat)
2. **Task 2: Run download and fix actual ZIP format issues** - `c8e4e7c` (fix)

## Files Created/Modified

- `configs/default.yaml` - Corrected figshare URL, added expected_md5, changed raw_dir to data_raw
- `requirements.txt` - Added requests>=2.31.0, imagehash==4.3.2, openpyxl>=3.1.0
- `scripts/download.py` - Full download pipeline replacing NotImplementedError placeholder
- `data_raw/dataset.csv` - 3,746 rows x 37 columns (converted from xlsx)
- `data_raw/images/` - 3,746 JPEG radiograph files (IMG000001.jpeg ... IMG003746.jpeg)
- `data_raw/Annotations/` - 1,867 LabelMe JSON annotation files (tumor images only)

## Decisions Made

- **xlsx-to-csv conversion:** The BTXRD ZIP ships `dataset.xlsx` (Excel), not `dataset.csv`. The download script converts it to CSV during extraction and removes the xlsx file. This ensures downstream scripts can use `pd.read_csv()` consistently. Added openpyxl dependency for the conversion.
- **Nested Annotations cleanup:** The ZIP contains a nested `Annotations/Annotations/` directory (duplicate of the top-level annotations). The script removes the nested copy after extraction.
- **Skip re-download for existing ZIP:** If `BTXRD.zip` already exists and MD5 matches, skip the download and proceed to extraction. This handles interrupted runs gracefully.
- **Logger naming:** Changed from `__name__` to `assurexray.download` so it inherits handlers from the `assurexray` parent logger created by `setup_logging()`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Dataset ships as xlsx, not csv**
- **Found during:** Task 2 (download execution)
- **Issue:** The BTXRD ZIP contains `dataset.xlsx` (Excel format), not `dataset.csv` as assumed by the plan and research. The research was based on the paper description which did not specify the file format.
- **Fix:** Added xlsx-to-csv conversion in `extract_and_organize()` using `pd.read_excel()` + `df.to_csv()`. Added `openpyxl>=3.1.0` to requirements.txt.
- **Files modified:** scripts/download.py, requirements.txt
- **Verification:** `data_raw/dataset.csv` exists with 3746 rows x 37 columns
- **Committed in:** c8e4e7c

**2. [Rule 3 - Blocking] Nested Annotations/Annotations/ directory**
- **Found during:** Task 2 (download execution)
- **Issue:** The ZIP extracts to `BTXRD/Annotations/` which itself contains both JSON files AND a nested `Annotations/` subdirectory with duplicate JSON files.
- **Fix:** Added cleanup logic in `extract_and_organize()` to remove the nested duplicate directory using `shutil.rmtree()`.
- **Files modified:** scripts/download.py
- **Verification:** `data_raw/Annotations/` contains 1867 JSON files and zero subdirectories
- **Committed in:** c8e4e7c

**3. [Rule 1 - Bug] Excel temp file in ZIP**
- **Found during:** Task 2 (download execution)
- **Issue:** The ZIP contains `~$dataset(total).xlsx`, an Excel temp/lock file that should not be present.
- **Fix:** Added glob cleanup for `~$*` files after extraction.
- **Files modified:** scripts/download.py
- **Verification:** No temp files in data_raw/
- **Committed in:** c8e4e7c

**4. [Rule 1 - Bug] Logger not producing output when run as script**
- **Found during:** Task 2 (verification)
- **Issue:** `logger = logging.getLogger(__name__)` creates a logger named `__main__` which has no handlers. The `setup_logging()` function creates handlers on the `assurexray` logger.
- **Fix:** Changed logger name to `assurexray.download` so it inherits from the `assurexray` parent logger.
- **Files modified:** scripts/download.py
- **Verification:** Running the script shows INFO-level log messages
- **Committed in:** c8e4e7c

---

**Total deviations:** 4 auto-fixed (2 blocking, 2 bugs)
**Impact on plan:** All fixes necessary for correct operation. No scope creep. The core download/verify/extract pipeline works as designed; fixes handle the actual ZIP contents which differed from paper assumptions.

## Issues Encountered

- The 801MB download took ~2 minutes on the connection, which exceeded the default command timeout. The download was handled successfully by using a longer timeout.

## User Setup Required

None - no external service configuration required. The virtual environment was created at `.venv/` with all dependencies installed.

## Dataset Column Reference (for Plan 02-02)

The actual 37 column names in `dataset.csv`:
```
image_id, center, age, gender, hand, ulna, radius, humerus, foot, tibia,
fibula, femur, hip bone, ankle-joint, knee-joint, hip-joint, wrist-joint,
elbow-joint, shoulder-joint, tumor, benign, malignant, osteochondroma,
multiple osteochondromas, simple bone cyst, giant cell tumor, osteofibroma,
synovial osteochondroma, other bt, osteosarcoma, other mt, upper limb,
lower limb, pelvis, frontal, lateral, oblique
```

Notable naming conventions:
- Spaces: `hip bone`, `multiple osteochondromas`, `simple bone cyst`, `giant cell tumor`, `synovial osteochondroma`, `other bt`, `other mt`, `upper limb`, `lower limb`
- Hyphens: `ankle-joint`, `knee-joint`, `hip-joint`, `wrist-joint`, `elbow-joint`, `shoulder-joint`
- Image filenames: `IMG000001.jpeg` through `IMG003746.jpeg` (`.jpeg` extension, not `.jpg`)

## Next Phase Readiness

- data_raw/ is fully populated and verified -- ready for Plan 02-02 (audit script and dataset spec)
- Column names are confirmed -- the audit script and dataset_spec.md can use exact names
- The `.jpeg` image extension is handled by the case-insensitive glob pattern
- No blockers for Phase 2 Plan 2

---
*Phase: 02-data-acquisition-and-audit*
*Completed: 2026-02-19*
