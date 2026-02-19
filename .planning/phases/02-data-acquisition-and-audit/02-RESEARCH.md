# Phase 2: Data Acquisition and Audit - Research

**Researched:** 2026-02-19
**Domain:** Dataset download from figshare, data profiling/auditing, markdown report generation
**Confidence:** HIGH (verified against figshare API, PMC paper, existing project scaffold)

## Summary

Phase 2 delivers three concrete outputs: (1) a download script that fetches the BTXRD dataset from figshare, verifies integrity, and organizes files into `data_raw/`, (2) an audit script that profiles the dataset and generates `docs/data_audit_report.md` with embedded figures, and (3) a hand-written `docs/dataset_spec.md` documenting all 37 columns of `dataset.csv`.

The BTXRD dataset is a single 801MB ZIP file (`BTXRD.zip`) hosted on figshare (article ID 27865398, file ID 50653575). The download URL uses a 302 redirect from `https://ndownloader.figshare.com/files/50653575` to an S3 presigned URL. The ZIP contains three items at the top level: an `images/` folder (3,746 JPEG radiographs), an `Annotations/` folder (LabelMe JSON files for tumor images only), and `dataset.csv` (37 columns, 3,746 rows). The figshare API provides a known MD5 checksum (`59132a5036d030580ccade8add8f13df`) for integrity verification.

The audit script needs to profile class distribution (confirming 1,879 Normal / 1,525 Benign / 342 Malignant), generate image dimension histograms, compute per-center breakdown (Center 1: 2,938, Center 2: 549, Center 3: 259), count missing values, check annotation coverage (matching JSON files to tumor images), and detect duplicate images. The leakage risk from same-lesion multi-angle images must be explicitly documented, along with available proxy grouping columns.

**Primary recommendation:** Use `requests` for streaming download with `tqdm` progress bar and `hashlib` MD5 verification. Use `pandas` + `matplotlib` + `seaborn` for the audit. Use `imagehash` (perceptual hashing) for duplicate detection. Generate figures as PNGs saved to `docs/figures/` and referenced in the markdown report. All libraries are already in `requirements.txt` or are stdlib except `imagehash` which should be added.

## Standard Stack

### Core (all already in requirements.txt unless noted)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| requests | (stdlib-adjacent, transitive dep) | HTTP download with streaming | Industry standard for HTTP; `urllib3` underneath handles redirects |
| hashlib | (stdlib) | MD5 checksum verification | Built-in, no extra dependency needed |
| zipfile | (stdlib) | ZIP extraction | Built-in, handles nested directory structures |
| tqdm | 4.67.1 | Download + processing progress bars | Already in requirements.txt |
| pandas | 2.2.3 | CSV loading, data profiling, missing value analysis | Already in requirements.txt |
| matplotlib | 3.10.0 | Figure generation (histograms, bar charts) | Already in requirements.txt |
| seaborn | 0.13.2 | Statistical plots (heatmaps, distribution plots) | Already in requirements.txt |
| Pillow | 11.1.0 | Image dimension reading (without loading full image) | Already in requirements.txt |
| json | (stdlib) | LabelMe annotation parsing | Built-in |
| pathlib | (stdlib) | Path manipulation | Built-in, preferred over os.path |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| imagehash | 4.3.2 | Perceptual image hashing for duplicate detection | Duplicate detection in audit script |
| shutil | (stdlib) | File/directory operations | Moving/organizing extracted files |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `requests` for download | `urllib.request.urlretrieve` | No streaming, no progress bar support, no redirect control |
| `imagehash` for duplicates | Raw pixel comparison (numpy) | Much slower, memory-intensive for 3,746 images; imagehash is O(1) per comparison |
| `imagehash` for duplicates | `imagededup` (idealo) | Full framework is overkill; imagehash is simpler for ~4K images |
| Manual markdown generation | ydata-profiling (auto HTML report) | ydata-profiling generates massive HTML, not the lean markdown we need. Our audit has specific sections (leakage risk, annotation coverage) that no auto-profiler covers |
| matplotlib for all plots | plotly | Interactive but generates HTML, not static PNGs for markdown embedding |

### New Dependency Required

**`imagehash==4.3.2`** must be added to `requirements.txt`. It depends on `Pillow`, `numpy`, and `scipy` -- all already present. This is the only new dependency for Phase 2.

**Note on `requests`:** The `requests` library is a transitive dependency of several packages already in `requirements.txt` (e.g., via `tqdm`, various pip internals). However, it is NOT explicitly listed. Two options:
1. Add `requests>=2.31.0` to `requirements.txt` explicitly (recommended for clarity)
2. Use `urllib.request` from stdlib instead (avoids adding a dependency but loses streaming + progress bar convenience)

**Recommendation:** Add `requests>=2.31.0` to `requirements.txt`. It is already installed in any Python environment with pip.

## Architecture Patterns

### Data Directory Structure After Download

```
data_raw/                          # Created by download.py
├── images/                        # 3,746 JPEG files
│   ├── {image_id}.jpg             # Naming convention TBD after download
│   └── ...
├── Annotations/                   # LabelMe JSON files (tumor images only)
│   ├── {image_id}.json            # One per tumor image (1,867 expected)
│   └── ...
└── dataset.csv                    # 37 columns, 3,746 rows

docs/                              # Output from audit.py + manual spec
├── figures/                       # Auto-generated by audit.py
│   ├── class_distribution.png
│   ├── dimension_histogram.png
│   ├── center_breakdown.png
│   ├── missing_values.png
│   ├── annotation_coverage.png
│   └── duplicate_detection.png
├── data_audit_report.md           # Auto-generated by audit.py
└── dataset_spec.md                # Hand-written (or generated template)
```

### Pattern 1: Streaming Download with Progress Bar and MD5 Verification

**What:** Download a large file using `requests` with `stream=True`, display progress via `tqdm`, compute MD5 incrementally during download, and verify against known checksum.

**When to use:** Any file download > 10MB where integrity matters.

**Example:**
```python
# Source: figshare API + requests docs + tqdm docs
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

FIGSHARE_FILE_URL = "https://ndownloader.figshare.com/files/50653575"
EXPECTED_MD5 = "59132a5036d030580ccade8add8f13df"
CHUNK_SIZE = 8192  # 8KB chunks

def download_with_verification(url: str, dest: Path, expected_md5: str) -> None:
    """Download file with progress bar and MD5 verification."""
    response = requests.get(url, stream=True, allow_redirects=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    md5_hash = hashlib.md5()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            md5_hash.update(chunk)
            pbar.update(len(chunk))

    actual_md5 = md5_hash.hexdigest()
    if actual_md5 != expected_md5:
        dest.unlink()  # Remove corrupt file
        raise ValueError(
            f"MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
```

**Key details:**
- `stream=True` prevents loading 801MB into memory
- `allow_redirects=True` handles the 302 from ndownloader to S3
- MD5 is computed incrementally during download (no second pass needed)
- `timeout=300` prevents hanging on network issues (5 min for 801MB)
- Content-length comes from the redirect target (S3), not the 302 response itself -- `requests` handles this automatically when following redirects

### Pattern 2: ZIP Extraction with Top-Level Stripping

**What:** Extract a ZIP file and handle the common case where the ZIP contains a top-level directory that wraps all content.

**When to use:** When the ZIP structure is `BTXRD/images/`, `BTXRD/Annotations/`, `BTXRD/dataset.csv` but you want `data_raw/images/`, `data_raw/Annotations/`, `data_raw/dataset.csv`.

**Example:**
```python
import zipfile
from pathlib import Path

def extract_and_organize(zip_path: Path, dest_dir: Path) -> None:
    """Extract ZIP and flatten single top-level directory if present."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check for common top-level wrapper directory
        top_levels = {name.split("/")[0] for name in zf.namelist() if "/" in name}
        has_wrapper = len(top_levels) == 1

        zf.extractall(dest_dir)

    # If there's a wrapper dir (e.g., BTXRD/), move contents up
    if has_wrapper:
        wrapper = dest_dir / top_levels.pop()
        for item in wrapper.iterdir():
            item.rename(dest_dir / item.name)
        wrapper.rmdir()
```

**Key details:**
- The BTXRD ZIP likely contains a top-level `BTXRD/` directory (common figshare pattern). The script must handle both cases (wrapper present or not).
- After extraction, verify expected structure: `images/` dir, `Annotations/` dir, `dataset.csv` file.

### Pattern 3: Audit Report as Markdown with Embedded PNGs

**What:** Generate matplotlib figures, save as PNGs, and write a markdown file referencing them with relative paths.

**When to use:** For the auto-generated `data_audit_report.md`.

**Example:**
```python
import matplotlib.pyplot as plt
from pathlib import Path

def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> str:
    """Save figure and return relative markdown path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Return path relative to docs/ (where the .md file lives)
    return f"figures/{name}.png"

def write_audit_report(sections: dict, output_path: Path) -> None:
    """Write markdown audit report with embedded figures."""
    with open(output_path, "w") as f:
        f.write("# BTXRD Data Audit Report\n\n")
        f.write(f"*Auto-generated by `scripts/audit.py`*\n\n")
        for title, content in sections.items():
            f.write(f"## {title}\n\n")
            f.write(content)
            f.write("\n\n")
```

**Key details:**
- Figures saved at 150 DPI for reasonable file size + quality balance
- `bbox_inches="tight"` prevents label clipping
- `facecolor="white"` ensures white background (not transparent)
- Relative paths in markdown (`figures/class_distribution.png`) so the report renders correctly on GitHub and locally

### Pattern 4: Perceptual Hash Duplicate Detection

**What:** Compute perceptual hashes for all images and find near-duplicates by comparing hash distances.

**When to use:** Duplicate detection in the audit script.

**Example:**
```python
import imagehash
from PIL import Image
from pathlib import Path
from collections import defaultdict

def find_duplicates(images_dir: Path, hash_size: int = 8, threshold: int = 5) -> list:
    """Find near-duplicate images using perceptual hashing.

    Args:
        images_dir: Directory containing images.
        hash_size: Hash grid size (8 = 64-bit hash).
        threshold: Maximum Hamming distance to consider as duplicate.

    Returns:
        List of (image_a, image_b, distance) tuples for near-duplicates.
    """
    hashes = {}
    for img_path in sorted(images_dir.glob("*.jpg")):
        img = Image.open(img_path)
        h = imagehash.phash(img, hash_size=hash_size)
        hashes[img_path.name] = h

    duplicates = []
    names = list(hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = hashes[names[i]] - hashes[names[j]]
            if dist <= threshold:
                duplicates.append((names[i], names[j], dist))

    return duplicates
```

**Key details:**
- `phash` (perceptual hash) is robust to resizing and minor compression differences
- Hash size 8 produces a 64-bit hash -- good balance of speed and accuracy
- Threshold of 5 catches near-duplicates while avoiding false positives (tunable)
- O(n^2) comparison is acceptable for 3,746 images (~7M comparisons, each is a simple integer subtract)
- For medical images, threshold may need to be lower (3-4) since legitimate different images can look similar

### Anti-Patterns to Avoid

- **Loading all images into memory for dimension check:** Use `PIL.Image.open().size` which reads only the header, not the full pixel data. Never load all 3,746 images as numpy arrays simultaneously.

- **Downloading without checksum verification:** The figshare API provides MD5 checksums. Always verify. A corrupt download wastes hours of debugging in later phases.

- **Hardcoding column names from the paper:** The paper's Table 3 shows example column names but the actual CSV may use slightly different naming (underscores, capitalization). Always read column names from the downloaded CSV, then verify against the expected 37-column schema.

- **Generating the audit report as HTML:** The requirement specifies markdown. HTML reports (like ydata-profiling output) are harder to version-control, review in PRs, and render consistently.

- **Re-downloading if data already exists:** The download script must check if `data_raw/` already has the expected files before downloading. Use file count + dataset.csv row count as a quick validation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image duplicate detection | Pixel-by-pixel comparison | `imagehash.phash()` | Perceptual hashing handles compression artifacts, resizing. Pixel comparison is O(n * pixels) vs O(n * 64 bits) |
| Progress bar for download | Print statements with byte counts | `tqdm` with `requests.iter_content()` | tqdm handles terminal width, ETA, speed display, units |
| MD5 checksum computation | Read entire file then hash | `hashlib.md5()` with incremental `.update()` | Memory-efficient for 801MB file |
| Image dimension reading | `numpy.array(Image.open(...))` | `Image.open(path).size` | `.size` reads JPEG header only (~100 bytes), not full pixel data |
| CSV data profiling | Manual loops over dataframe | `pandas.DataFrame.describe()`, `.isna().sum()`, `.value_counts()` | Pandas has optimized C implementations |
| Bar charts for audit | Manual matplotlib axis setup | `seaborn.countplot()` / `seaborn.barplot()` | Seaborn handles aesthetics, labels, colors automatically |

**Key insight:** Phase 2 is data plumbing, not ML. Every minute spent building custom utilities for download, hashing, or profiling is wasted. Use pandas idioms and established libraries exactly as documented.

## Common Pitfalls

### Pitfall 1: Figshare Download URL Returns 302 Redirect

**What goes wrong:** Using `urllib.request.urlretrieve()` or `requests.get()` without `allow_redirects=True` gets a 0-byte file or an HTML error page instead of the ZIP.

**Why it happens:** The figshare download URL (`https://ndownloader.figshare.com/files/50653575`) returns an HTTP 302 redirect to a time-limited S3 presigned URL. The presigned URL expires after ~10 seconds.

**How to avoid:** Use `requests.get(url, stream=True, allow_redirects=True)`. The `requests` library follows redirects by default, but make it explicit. Do NOT try to cache or reuse the presigned S3 URL -- it expires quickly.

**Warning signs:** Downloaded file is 0 bytes, or file is HTML/text instead of ZIP. `zipfile.is_zipfile()` returns False on the downloaded file.

### Pitfall 2: ZIP Contains Top-Level Wrapper Directory

**What goes wrong:** After `extractall()`, the data is at `data_raw/BTXRD/images/` instead of `data_raw/images/`. Subsequent scripts fail because they look for `data_raw/images/`.

**Why it happens:** Figshare dataset ZIPs commonly wrap all files in a directory matching the ZIP filename.

**How to avoid:** After extraction, check if a single top-level directory exists. If so, move its contents up one level and remove the empty wrapper. Verify the expected structure (`images/`, `Annotations/`, `dataset.csv`) exists at the expected level.

**Warning signs:** `FileNotFoundError` when scripts look for `data_raw/images/` or `data_raw/dataset.csv`.

### Pitfall 3: Config URL Points to Article, Not File

**What goes wrong:** The current `configs/default.yaml` has `figshare_url: "https://figshare.com/ndownloader/articles/21899338/versions/1"` which is (a) the WRONG article ID (21899338 instead of 27865398) and (b) an article-level URL that returns HTML/challenge, not a ZIP file.

**Why it happens:** The URL was set as a placeholder in Phase 1 before the actual figshare data was researched.

**How to avoid:** Update the config to use the direct file download URL: `https://ndownloader.figshare.com/files/50653575`. Alternatively, use the figshare API programmatically: `GET https://api.figshare.com/v2/articles/27865398/files` returns the file's `download_url`.

**Warning signs:** Download returns HTML, 502 error, or WAF challenge instead of binary ZIP data.

### Pitfall 4: Image File Count Mismatch

**What goes wrong:** The audit reports fewer or more images than 3,746 because some images have non-`.jpg` extensions, or hidden files (`.DS_Store`, `Thumbs.db`) are counted as images.

**Why it happens:** Glob patterns like `*` match everything. The dataset description says JPEG but does not specify the exact extension case (`.jpg` vs `.jpeg` vs `.JPG`).

**How to avoid:** Use case-insensitive glob: `images_dir.glob("*.[jJ][pP][gG]")` or filter by common JPEG extensions (`.jpg`, `.jpeg`, `.JPG`, `.JPEG`). Explicitly exclude hidden files.

**Warning signs:** Image count in audit does not match 3,746.

### Pitfall 5: Annotation Coverage Assumes 1:1 Image-Annotation Mapping

**What goes wrong:** The audit incorrectly reports annotation coverage by assuming every image has a corresponding JSON annotation. In reality, only tumor images (1,867) have annotations. Normal images (1,879) have no annotations.

**Why it happens:** Misunderstanding the dataset structure: annotations are only created for images containing tumors.

**How to avoid:** Compute annotation coverage as: `(number of JSON files in Annotations/) / (number of tumor images from dataset.csv where tumor=1)`. Expected: ~100% of tumor images have annotations, 0% of normal images.

**Warning signs:** Annotation coverage reported as ~50% when it should be ~100% of tumor images.

### Pitfall 6: Column Name Assumptions from Paper vs Actual CSV

**What goes wrong:** The dataset_spec.md documents column names from the paper's Table 3, but the actual CSV uses different naming conventions (e.g., `hip_bone` vs `hip-bone`, spaces in column names, different capitalization).

**Why it happens:** Papers often format column names for readability. The actual CSV may use underscores, hyphens, or spaces.

**How to avoid:** The dataset_spec.md must be written AFTER downloading the data and inspecting the actual CSV headers. The audit script should print actual column names. Do not pre-write the spec from the paper alone.

**Warning signs:** `KeyError` when accessing columns by name in later phases.

### Pitfall 7: License Discrepancy Between Figshare API and Paper

**What goes wrong:** The figshare API metadata reports `CC BY 4.0` but the paper explicitly states `CC BY-NC-ND 4.0`. Using the wrong license in documentation could have legal implications.

**Why it happens:** Figshare's metadata was set by the uploader and may not match the paper's stated license. The paper is the authoritative source.

**How to avoid:** Document the license as **CC BY-NC-ND 4.0** (from the paper). Note the figshare metadata discrepancy in the dataset_spec.md for transparency.

**Warning signs:** None -- this is a documentation issue, not a runtime error.

## Code Examples

### Complete Download Pattern with Figshare API Discovery

```python
# Source: figshare API docs (https://docs.figshare.com/)
# Verified: 2026-02-19 via curl against live API

import requests
import hashlib
import zipfile
from pathlib import Path
from tqdm import tqdm

FIGSHARE_ARTICLE_ID = 27865398
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"

def get_download_info() -> dict:
    """Get file download URL and MD5 from figshare API."""
    resp = requests.get(FIGSHARE_API_URL, timeout=30)
    resp.raise_for_status()
    files = resp.json()
    assert len(files) == 1, f"Expected 1 file, got {len(files)}"
    return {
        "url": files[0]["download_url"],       # https://ndownloader.figshare.com/files/50653575
        "md5": files[0]["computed_md5"],        # 59132a5036d030580ccade8add8f13df
        "size": files[0]["size"],               # 840474929 bytes (~801 MB)
        "name": files[0]["name"],               # BTXRD.zip
    }

def download_file(url: str, dest: Path, expected_md5: str) -> None:
    """Stream-download with tqdm progress and MD5 verification."""
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    md5 = hashlib.md5()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            md5.update(chunk)
            bar.update(len(chunk))

    if md5.hexdigest() != expected_md5:
        dest.unlink()
        raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {md5.hexdigest()}")

def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract ZIP, handle top-level wrapper directory."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        top_dirs = {n.split("/")[0] for n in zf.namelist() if "/" in n}
        zf.extractall(dest)

    # Flatten single wrapper directory if present
    if len(top_dirs) == 1:
        wrapper = dest / top_dirs.pop()
        if wrapper.is_dir():
            for item in wrapper.iterdir():
                item.rename(dest / item.name)
            wrapper.rmdir()

def verify_structure(raw_dir: Path) -> None:
    """Verify expected data structure after extraction."""
    images_dir = raw_dir / "images"
    annot_dir = raw_dir / "Annotations"
    csv_path = raw_dir / "dataset.csv"

    assert images_dir.is_dir(), f"Missing images directory: {images_dir}"
    assert annot_dir.is_dir(), f"Missing Annotations directory: {annot_dir}"
    assert csv_path.is_file(), f"Missing dataset.csv: {csv_path}"

    n_images = len(list(images_dir.glob("*.[jJ][pP][gG]")) +
                    list(images_dir.glob("*.[jJ][pP][eE][gG]")))
    print(f"Images found: {n_images}")
    print(f"Annotations found: {len(list(annot_dir.glob('*.json')))}")
```

### Audit Script Sections

```python
# Source: pandas + matplotlib + seaborn standard API patterns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

def audit_class_distribution(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Generate class distribution bar chart and summary text."""
    # Derive 3-class label
    def get_label(row):
        if row["malignant"] == 1:
            return "Malignant"
        elif row["benign"] == 1:
            return "Benign"
        else:
            return "Normal"

    df["label"] = df.apply(get_label, axis=1)
    counts = df["label"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # green, blue, red
    counts.reindex(["Normal", "Benign", "Malignant"]).plot.bar(
        ax=ax, color=colors, edgecolor="white"
    )
    ax.set_title("Class Distribution", fontsize=14)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.reindex(["Normal", "Benign", "Malignant"])):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")

    summary = (
        f"| Class | Count | Percentage |\n"
        f"|-------|-------|------------|\n"
        f"| Normal | {counts.get('Normal', 0)} | {counts.get('Normal', 0)/len(df)*100:.1f}% |\n"
        f"| Benign | {counts.get('Benign', 0)} | {counts.get('Benign', 0)/len(df)*100:.1f}% |\n"
        f"| Malignant | {counts.get('Malignant', 0)} | {counts.get('Malignant', 0)/len(df)*100:.1f}% |\n"
        f"| **Total** | **{len(df)}** | **100%** |\n"
    )
    return fig, summary

def audit_image_dimensions(images_dir: Path) -> tuple[plt.Figure, str]:
    """Generate image dimension histogram."""
    widths, heights = [], []
    for img_path in sorted(images_dir.glob("*.[jJ][pP][gG]")):
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=50, edgecolor="white")
    axes[0].set_title("Image Widths")
    axes[0].set_xlabel("Pixels")
    axes[1].hist(heights, bins=50, edgecolor="white")
    axes[1].set_title("Image Heights")
    axes[1].set_xlabel("Pixels")
    fig.suptitle("Image Dimension Distribution", fontsize=14)
    fig.tight_layout()

    summary = (
        f"- Width range: {min(widths)} - {max(widths)} pixels\n"
        f"- Height range: {min(heights)} - {max(heights)} pixels\n"
        f"- Total images measured: {len(widths)}\n"
    )
    return fig, summary

def audit_missing_values(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Generate missing value analysis."""
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, 6))
    missing[missing > 0].sort_values(ascending=True).plot.barh(ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Count")

    summary = "| Column | Missing | Percentage |\n|--------|---------|------------|\n"
    for col in df.columns:
        if missing[col] > 0:
            summary += f"| {col} | {missing[col]} | {missing_pct[col]}% |\n"
    if missing.sum() == 0:
        summary += "| *(none)* | 0 | 0% |\n"

    return fig, summary
```

### Per-Center Breakdown Pattern

```python
def audit_center_breakdown(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Generate per-center class distribution breakdown."""
    cross = pd.crosstab(df["center"], df["label"])
    cross = cross.reindex(columns=["Normal", "Benign", "Malignant"])

    fig, ax = plt.subplots(figsize=(8, 5))
    cross.plot.bar(ax=ax, stacked=True, color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.set_title("Class Distribution by Center")
    ax.set_ylabel("Count")
    ax.legend(title="Class")
    fig.tight_layout()

    summary = cross.to_markdown() + "\n"
    return fig, summary
```

### Annotation Coverage Analysis

```python
def audit_annotation_coverage(
    df: pd.DataFrame, images_dir: Path, annot_dir: Path
) -> str:
    """Check annotation coverage: tumor images should have JSON files."""
    tumor_images = set(df[df["tumor"] == 1]["image_id"].values)
    json_files = {p.stem for p in annot_dir.glob("*.json")}

    covered = tumor_images & json_files
    missing_annotations = tumor_images - json_files
    extra_annotations = json_files - tumor_images

    summary = (
        f"- Tumor images in dataset.csv: {len(tumor_images)}\n"
        f"- JSON annotation files found: {len(json_files)}\n"
        f"- Tumor images with annotations: {len(covered)} "
        f"({len(covered)/len(tumor_images)*100:.1f}%)\n"
        f"- Tumor images missing annotations: {len(missing_annotations)}\n"
        f"- Extra annotation files (no matching tumor image): {len(extra_annotations)}\n"
    )
    return summary
```

### Leakage Risk Documentation Template

```markdown
## Data Leakage Risk Assessment

### Same-Lesion Multi-Angle Images

The BTXRD dataset includes radiographs taken from multiple shooting angles
(frontal, lateral, oblique) for the same patient visit. Since **no patient_id
column exists** in dataset.csv, there is no way to definitively group images
by patient.

**Risk:** If images of the same lesion from different angles are split across
train and test sets, the model may learn patient-specific features (bone shape,
implant presence) rather than tumor characteristics, inflating test performance.

### Available Proxy Grouping Columns

The following columns could be used as a proxy for patient grouping, but
each has limitations:

| Column Combination | Uniqueness | Limitation |
|--------------------|------------|------------|
| center + age + gender | Low | Many patients share demographics |
| center + age + gender + anatomical_site | Medium | Better but not unique |
| center + age + gender + anatomical_site + shooting_angle | Higher | Different angles = same patient |

**Recommendation (from REQUIREMENTS.md):** Do NOT fabricate patient groupings.
Acknowledge the limitation honestly in the audit report. The center-holdout
split (Phase 3) partially mitigates this by testing on entirely different
data sources.
```

## Dataset.csv Column Specification (37 Columns)

Based on the paper (PMC11739492, Table 3), the 37 columns are:

### Identity and Demographics (4 columns)
1. `image_id` -- Unique image identifier
2. `center` -- Source center (1, 2, or 3)
3. `age` -- Patient age in years (range: 1-88; mean: 37.53 +/- 19.58)
4. `gender` -- Patient gender (likely: male/female or M/F)

### Anatomical Site (15 columns, binary 0/1)
5. `hand`
6. `ulna`
7. `radius`
8. `humerus`
9. `foot`
10. `tibia`
11. `fibula`
12. `femur`
13. `hip-bone` (or `hip_bone`)
14. `ankle-joint` (or `ankle_joint`)
15. `knee-joint` (or `knee_joint`)
16. `hip-joint` (or `hip_joint`)
17. `wrist-joint` (or `wrist_joint`)
18. `elbow-joint` (or `elbow_joint`)
19. `shoulder-joint` (or `shoulder_joint`)

### Classification Labels (3 columns, binary 0/1)
20. `tumor` -- 1 if any tumor present, 0 if normal
21. `benign` -- 1 if benign tumor
22. `malignant` -- 1 if malignant tumor

### Tumor Subtypes (9 columns, binary 0/1)
23. `osteochondroma` (754 cases)
24. `multiple osteochondromas` (or `multiple_osteochondromas`) (263 cases)
25. `simple bone cyst` (or `simple_bone_cyst`) (206 cases)
26. `giant cell tumor` (or `giant_cell_tumor`) (count TBD)
27. `osteofibroma` (count TBD)
28. `synovial osteochondroma` (or `synovial_osteochondroma`) (count TBD)
29. `other bt` (or `other_bt`) -- other benign tumors (count TBD)
30. `osteosarcoma` (297 cases)
31. `other mt` (or `other_mt`) -- other malignant tumors (count TBD)

### Body Region (3 columns, binary 0/1)
32. `upper limb` (or `upper_limb`)
33. `lower limb` (or `lower_limb`)
34. `pelvis`

### Shooting Angle (3 columns, binary 0/1)
35. `frontal` (2,181 images)
36. `lateral` (1,269 images)
37. `oblique` (296 images)

**CRITICAL NOTE:** The exact column names (hyphens vs underscores vs spaces) must be confirmed after download by reading the actual CSV header. The paper shows formatted names that may differ from the raw CSV. The audit script should print `df.columns.tolist()` as its first action and the dataset_spec.md must use the actual names.

### Label Derivation Logic

```python
# 3-class label derivation (from paper):
# Priority order: malignant > benign > normal
if row["malignant"] == 1:
    label = "Malignant"
elif row["benign"] == 1:
    label = "Benign"
elif row["tumor"] == 0:
    label = "Normal"
# Note: tumor=1 && benign=0 && malignant=0 should not exist
# (all tumor images are either benign or malignant)
```

### Per-Center Provenance

| Center | Source | Normal | Benign | Malignant | Total | % of Dataset |
|--------|--------|--------|--------|-----------|-------|--------------|
| 1 | Three Chinese hospitals (Guangxi Medical University, etc.) | 1,593 | 1,110 | 235 | 2,938 | 78.4% |
| 2 | Radiopaedia.org (online open-access) | 259 | 214 | 76 | 549 | 14.7% |
| 3 | MedPix database (US National Library of Medicine) | 27 | 201 | 31 | 259 | 6.9% |
| **Total** | | **1,879** | **1,525** | **342** | **3,746** | **100%** |

## Config Updates Required

The current `configs/default.yaml` has an incorrect figshare URL:
```yaml
# CURRENT (WRONG -- wrong article ID, article-level URL returns HTML):
figshare_url: "https://figshare.com/ndownloader/articles/21899338/versions/1"

# CORRECTED (direct file download URL):
figshare_url: "https://ndownloader.figshare.com/files/50653575"
```

Additional config entries that should be added for Phase 2:
```yaml
data:
  figshare_url: "https://ndownloader.figshare.com/files/50653575"
  figshare_article_id: 27865398
  figshare_file_id: 50653575
  expected_md5: "59132a5036d030580ccade8add8f13df"
  expected_image_count: 3746
  raw_dir: data_raw   # Note: current config says "data/raw" -- should be "data_raw" per success criteria
```

**Important:** The success criteria specify `data_raw/images/`, `data_raw/annotations/`, and `data_raw/dataset.csv`. The current config has `raw_dir: data/raw`. This should be changed to `data_raw` to match the success criteria and `.gitignore` pattern.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pandas-profiling (auto HTML report) | ydata-profiling 4.x or custom pandas audit | 2023 (rename + AGPL licensing) | Custom audit is better for this project -- specific sections needed |
| `urllib.request.urlretrieve()` | `requests.get(stream=True)` | Long-standing | Streaming + progress + redirect handling in one call |
| Full pixel comparison for duplicates | Perceptual hashing (`imagehash`) | Standard since 2018 | O(n) hashing + O(n^2) integer comparison vs O(n^2 * pixels) |
| Manual figure embedding in reports | matplotlib `savefig()` + markdown references | Standard pattern | Clean separation of figure generation and report writing |

**Deprecated/outdated:**
- `pandas-profiling`: Renamed to `ydata-profiling`, AGPL license. Do not use for this project -- custom audit is more appropriate.
- `urllib.request.urlretrieve()`: No streaming, no progress bar, poor redirect handling. Use `requests` instead.

## Open Questions

1. **Exact column names in dataset.csv**
   - What we know: Paper Table 3 lists 37 columns with formatted names (may have spaces or hyphens)
   - What's unclear: Whether the CSV uses `hip-bone`, `hip_bone`, or `hip bone` (and similarly for all multi-word columns)
   - Recommendation: The download script should run first. The audit script's first action should be printing `df.columns.tolist()`. The dataset_spec.md must use actual column names, not paper names.

2. **Exact annotation file naming convention**
   - What we know: Annotations are in LabelMe JSON format in `Annotations/` directory
   - What's unclear: Whether annotation filenames match image_id exactly (e.g., `image_001.json` for `image_001.jpg`) or use a different naming convention
   - Recommendation: The audit script should verify the mapping between image filenames and annotation filenames.

3. **Whether `data_raw/` folder name in ZIP contains `Annotations` (capital A) or `annotations`**
   - What we know: The figshare description says "Annotations" (capital A)
   - What's unclear: The success criteria say `data_raw/annotations/` (lowercase)
   - Recommendation: After extraction, check the actual case and either rename or update the success criteria reference. Use the actual case from the ZIP.

4. **Whether `imagehash` is a justified new dependency**
   - What we know: Duplicate detection is required by DATA-02. `imagehash` is lightweight (depends on Pillow, numpy, scipy -- all already present).
   - What's unclear: Whether a simpler approach (file size + MD5 hash for exact duplicates only) is sufficient.
   - Recommendation: Use `imagehash` for near-duplicate detection (perceptual similarity). Also compute MD5 for exact duplicate detection. This covers both cases. The dependency is justified.

5. **Raw directory naming: `data_raw` vs `data/raw`**
   - What we know: Success criteria say `data_raw/images/`. Current config says `raw_dir: data/raw`. `.gitignore` has `data_raw/`.
   - What's unclear: Which is authoritative.
   - Recommendation: Use `data_raw` (flat, matches success criteria and .gitignore). Update the config.

## Sources

### Primary (HIGH confidence)
- [Figshare API v2 -- articles/27865398/files](https://api.figshare.com/v2/articles/27865398/files) -- Verified file ID, download URL, MD5, size (curl on 2026-02-19)
- [Figshare API v2 -- articles/27865398](https://api.figshare.com/v2/articles/27865398) -- Article metadata, license info (curl on 2026-02-19)
- [PMC11739492](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739492/) -- Full paper with Table 3 (37 columns), per-center breakdown, class distribution, annotation details
- [figshare download URL headers](https://ndownloader.figshare.com/files/50653575) -- Verified 302 redirect to S3 presigned URL (curl -I on 2026-02-19)
- Python stdlib docs: `zipfile`, `hashlib`, `pathlib`, `json` -- Standard library, stable API

### Secondary (MEDIUM confidence)
- [ImageHash 4.3.2 on PyPI](https://pypi.org/project/ImageHash/) -- Current version verified, dependencies confirmed
- [LabelMe JSON format (roboflow.com)](https://roboflow.com/formats/labelme-json) -- JSON structure fields documented
- [requests + tqdm download pattern (GitHub gist)](https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51) -- Widely referenced streaming download pattern
- [BTXRD GitHub repo](https://github.com/SHUNHANYAO/BTXRD) -- Classification task code, model weights

### Tertiary (LOW confidence)
- Column name formatting (hyphens vs underscores vs spaces): Must be confirmed after actual download. Paper Table 3 formatting may not match CSV headers.
- Annotation filename convention: Must be confirmed after actual download.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries already in requirements.txt (except imagehash). Download URL and MD5 verified via live API call.
- Architecture: HIGH -- Download/extract/audit patterns are well-established. Dataset structure confirmed from paper and API.
- Pitfalls: HIGH -- Verified the figshare redirect behavior, confirmed config URL is wrong, identified license discrepancy with evidence.
- Column specification: MEDIUM -- Based on paper Table 3, but exact CSV header names need confirmation after download.

**Research date:** 2026-02-19
**Valid until:** 2026-04-19 (60 days -- figshare URLs and dataset are stable; no expected changes)
