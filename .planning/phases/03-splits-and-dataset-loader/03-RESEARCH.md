# Phase 3: Splits and Dataset Loader - Research

**Researched:** 2026-02-19
**Domain:** Data splitting strategies, PyTorch Dataset/DataLoader, albumentations augmentation pipelines
**Confidence:** HIGH

## Summary

Phase 3 delivers three interconnected components: (1) a split script (`scripts/split.py`) that generates reproducible CSV manifests for two splitting strategies (stratified 70/15/15 and center-holdout), (2) augmentation pipelines in `src/data/transforms.py` using albumentations for train/val/test modes, and (3) a PyTorch Dataset class in `src/data/dataset.py` that loads images from split manifests and returns correctly shaped tensors. The split utilities live in `src/data/split_utils.py`.

All required libraries are already in `requirements.txt`: scikit-learn 1.7.2 (for `train_test_split` with `stratify`), albumentations 2.0.8 (for CLAHE, Resize, HorizontalFlip, Rotate, Normalize), Pillow 11.1.0 (image loading), and torch 2.6.0 / torchvision 0.21.0 (Dataset, DataLoader). No new dependencies are needed.

The most critical implementation decisions are: (1) handling the 21 exact duplicate image pairs discovered in Phase 2 by assigning duplicate groups to the same split side, (2) using `train_test_split` with `stratify` parameter for the two-step 70/15/15 stratified split (first split 70/30, then split the 30 into 50/50 for val/test), (3) applying CLAHE with `p=1.0` (always) in training augmentation as it is a preprocessing step for radiographs rather than a random augmentation, and (4) using PIL/Pillow for image loading (converting to numpy for albumentations) rather than OpenCV to avoid BGR/RGB pitfalls.

**Primary recommendation:** Use scikit-learn `train_test_split` with `stratify` for the stratified split (two-step: train vs rest, then val vs test). Use pandas filtering on the `center` column for the center-holdout split. Build a single `BTXRDDataset` class that accepts a split CSV and a transform pipeline, loading images lazily with PIL and converting to numpy arrays for albumentations.

## Standard Stack

### Core (all already in requirements.txt)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.7.2 | `train_test_split` with stratify parameter | Industry-standard stratified splitting with deterministic random_state |
| albumentations | 2.0.8 | CLAHE, Resize, HorizontalFlip, Rotate, Normalize, Compose | Fast numpy-based augmentations, MIT license, CLAHE not available in torchvision |
| torch | 2.6.0 | `Dataset`, `DataLoader` base classes | Pinned in Phase 1 |
| torchvision | 0.21.0 | Not directly used in transforms (albumentations handles all) but available | Pinned in Phase 1 |
| pandas | 2.2.3 | CSV manifest read/write, DataFrame filtering for center split | Already in requirements.txt |
| Pillow | 11.1.0 | Image loading via `Image.open()` | Already in requirements.txt, natively produces RGB |
| numpy | 2.2.3 | Array conversion for albumentations input | Already in requirements.txt |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| imagehash | 4.3.2 | Recompute perceptual hashes for duplicate grouping | Only in split script to identify duplicate pairs |
| cv2 (opencv) | transitive via albumentations | CLAHE internal implementation, interpolation | Only used internally by albumentations; not imported directly |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `train_test_split` (two-step) | `StratifiedShuffleSplit` | StratifiedShuffleSplit is designed for k-fold CV, train_test_split is simpler for a single split; two-step approach is explicit and easy to verify |
| PIL for image loading | cv2.imread + cvtColor | OpenCV requires BGR-to-RGB conversion which is a common bug source; PIL natively loads as RGB |
| albumentations Normalize | torchvision.transforms.Normalize | Would require mixing albumentations and torchvision pipelines; keeping everything in albumentations is cleaner |
| Hardcoded duplicate list | Re-computing phash at split time | Hardcoded list is fragile to data changes; recomputing is ~30 seconds and guarantees correctness |

**Installation:** No new packages needed. All are in `requirements.txt`.

## Architecture Patterns

### Recommended Project Structure

```
scripts/
  split.py                    # CLI entry point: loads config, calls split_utils, writes CSVs

src/data/
  split_utils.py              # Pure functions: stratified_split(), center_holdout_split()
  transforms.py               # get_train_transforms(), get_val_transforms(), get_test_transforms()
  dataset.py                  # BTXRDDataset(Dataset) class
  __init__.py                 # Exports: BTXRDDataset, get_transforms

data/splits/                  # Output directory (CSV manifests)
  stratified_train.csv
  stratified_val.csv
  stratified_test.csv
  center_train.csv
  center_val.csv
  center_test.csv
```

### Pattern 1: Two-Step Stratified Split (70/15/15)

**What:** Use `train_test_split` twice to get a three-way split. First split into train (70%) and remainder (30%), then split remainder into val (50% of 30% = 15%) and test (50% of 30% = 15%). Both calls use `stratify` to preserve class proportions.

**When to use:** Any time you need a 3-way stratified split with scikit-learn.

**Why two-step:** scikit-learn `train_test_split` only produces two groups. There is no built-in 3-way stratified split function.

**Example:**
```python
# Source: scikit-learn 1.8.0 train_test_split docs
from sklearn.model_selection import train_test_split

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test with class stratification."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Step 1: train vs (val + test)
    remainder_ratio = val_ratio + test_ratio  # 0.30
    df_train, df_remainder = train_test_split(
        df,
        test_size=remainder_ratio,
        stratify=df[label_col],
        random_state=seed,
    )

    # Step 2: val vs test (from remainder)
    val_fraction_of_remainder = val_ratio / remainder_ratio  # 0.5
    df_val, df_test = train_test_split(
        df_remainder,
        test_size=1.0 - val_fraction_of_remainder,  # 0.5
        stratify=df_remainder[label_col],
        random_state=seed,
    )

    return df_train, df_val, df_test
```

### Pattern 2: Center-Holdout Split

**What:** Filter by `center` column. Center 1 images become train+val (split with stratification). Centers 2+3 become test. No random element in the train/test boundary -- it is deterministic by center assignment.

**When to use:** Testing geographic/institutional generalization. The center-holdout split is the more clinically meaningful evaluation.

**Example:**
```python
def center_holdout_split(
    df: pd.DataFrame,
    train_centers: list[int] = [1],
    test_centers: list[int] = [2, 3],
    val_ratio: float = 0.15,
    seed: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by center: train_centers for train/val, test_centers for test."""
    df_trainval = df[df["center"].isin(train_centers)].copy()
    df_test = df[df["center"].isin(test_centers)].copy()

    # Stratified train/val within train centers
    # val_ratio is relative to total, but here it is relative to trainval
    # Center 1 has 2938 images. 15% of 3746 = 562 val images.
    # 562 / 2938 = ~0.191 of Center 1
    val_fraction = val_ratio / (1.0 - (len(df_test) / len(df)))

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_fraction,
        stratify=df_trainval[label_col],
        random_state=seed,
    )

    return df_train, df_val, df_test
```

### Pattern 3: Duplicate-Aware Splitting

**What:** Before splitting, assign each image to a "group" based on perceptual hash. Exact duplicates (phash distance 0) get the same group ID. When performing the split, ensure all images in the same group land on the same side.

**Implementation approach:** Since scikit-learn `train_test_split` does not natively support group constraints, and the duplicate count is small (21 pairs = 42 images out of 3,746), the practical approach is:
1. Compute phash groups at split time using `imagehash`
2. For each duplicate pair, pick one "representative" image for the split allocation
3. After the split, force the partner image into the same split as its representative
4. Verify that class proportions are not meaningfully disturbed (they will not be, given only 42 images)

**Alternative simpler approach (recommended):** Since the duplicate pairs are known and small in number, the split script can:
1. Compute phash groups
2. Remove one image from each exact duplicate pair before splitting
3. After splitting, add the removed duplicates back to the same split as their partner
4. Log which duplicates were grouped together

### Pattern 4: Albumentations Compose Pipeline

**What:** Separate transform pipelines for train/val/test. Train gets the full augmentation chain. Val and test get only deterministic preprocessing.

**Example:**
```python
# Source: albumentations 2.0.8 docs, PyTorch classification example
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Full augmentation pipeline for training."""
    return A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, fill=0, p=0.5),
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic preprocessing for validation."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_test_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic preprocessing for testing (same as val)."""
    return get_val_transforms(image_size)
```

### Pattern 5: PyTorch Dataset with Albumentations

**What:** A Dataset class that reads a split CSV, lazily loads images with PIL, converts to numpy, applies albumentations transforms, and returns `(tensor, label_index)` tuples.

**Example:**
```python
# Source: PyTorch Dataset tutorial + albumentations classification example
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

CLASS_TO_IDX = {"Normal": 0, "Benign": 1, "Malignant": 2}

class BTXRDDataset(Dataset):
    """BTXRD bone tumor radiograph dataset."""

    def __init__(
        self,
        manifest_csv: str | Path,
        images_dir: str | Path,
        transform: A.Compose | None = None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Validate required columns
        assert "image_id" in self.df.columns, "manifest must have image_id column"
        assert "label" in self.df.columns, "manifest must have label column"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label_str = row["label"]
        label_idx = CLASS_TO_IDX[label_str]

        # Load image with PIL (natively RGB)
        image_path = self.images_dir / image_id
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)  # HWC uint8 numpy array

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]  # Now a torch.Tensor (CHW)

        return image, label_idx
```

### Pattern 6: CSV Manifest Format

**What:** Each split CSV contains three columns: `image_id`, `split`, and `label`. The `image_id` is the filename (e.g., `IMG000001.jpeg`), `split` is `train`/`val`/`test`, and `label` is `Normal`/`Benign`/`Malignant`.

**CSV example:**
```csv
image_id,split,label
IMG000001.jpeg,train,Normal
IMG000002.jpeg,train,Benign
IMG000342.jpeg,val,Malignant
```

**Key decisions:**
- `image_id` uses the actual filename with extension (matching the `image_id` column in `dataset.csv` which includes `.jpeg` or `.jpg`)
- `label` uses string names, not integers (human-readable, less error-prone)
- Each strategy gets its own set of 3 files (not one file with a strategy column)
- CSVs are committed to version control for reproducibility

### Anti-Patterns to Avoid

- **Splitting by index instead of by content:** Never split by row number. Always use stratification on the label column.
- **Forgetting to set random_state:** Every `train_test_split` call must pass `random_state=seed` for reproducibility.
- **Computing normalization stats from full dataset:** Use ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Do NOT compute mean/std from the training set -- ImageNet stats are standard for transfer learning.
- **Applying augmentation during validation/test:** Val/test transforms must be deterministic (Resize + Normalize + ToTensorV2 only).
- **Loading all images into memory in __init__:** Load lazily in `__getitem__`. The dataset has 3,746 images of variable size -- loading all at once is wasteful.
- **Ignoring mixed extensions:** 3,719 images use `.jpeg`, 27 use `.jpg`. The `image_id` column in `dataset.csv` already includes the correct extension. Use it as-is.
- **Duplicates crossing split boundaries:** The 21 exact duplicate pairs must be on the same side of any split.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stratified splitting | Custom random sampling with proportion checking | `sklearn.model_selection.train_test_split(stratify=...)` | Handles edge cases with small classes (342 Malignant), ensures exact proportions, deterministic with random_state |
| CLAHE preprocessing | Manual OpenCV `cv2.createCLAHE()` with LAB conversion | `albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(8,8))` | Handles RGB-to-LAB-to-RGB internally, integrates with Compose pipeline, tested with uint8 images |
| Image normalization | Manual `(image / 255.0 - mean) / std` | `albumentations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)` | Handles `max_pixel_value=255.0` correctly, integrates with pipeline, avoids float precision bugs |
| Numpy to tensor conversion | Manual `torch.from_numpy(image.transpose(2,0,1))` | `albumentations.pytorch.ToTensorV2()` | Handles HWC-to-CHW, float conversion, edge cases with grayscale |
| Duplicate detection at split time | Manually listing 21 known pairs | Recompute via `imagehash.phash()` at split time | Robust to data changes, self-documenting, only ~30s on 3,746 images |
| Label derivation | If/else chains in dataset __getitem__ | Pre-derive labels in split script, store in CSV manifest | Single source of truth, no logic duplication, verifiable offline |

**Key insight:** The augmentation pipeline involves subtle interactions (uint8 vs float32, HWC vs CHW, RGB vs BGR, ImageNet stats). Using albumentations' built-in pipeline handles all these correctly. Hand-rolling any piece risks silent correctness bugs that are extremely hard to detect (model trains but learns wrong features).

## Common Pitfalls

### Pitfall 1: CLAHE on Already-Normalized Images

**What goes wrong:** Applying CLAHE after normalization (float32, zero-centered) instead of before (uint8, 0-255 range). CLAHE expects uint8 input.
**Why it happens:** People put Normalize before CLAHE in the pipeline.
**How to avoid:** CLAHE must come BEFORE Normalize in the A.Compose pipeline. Order: CLAHE -> augmentations -> Resize -> Normalize -> ToTensorV2.
**Warning signs:** Model trains but accuracy is poor; pixel values look wrong in debug visualization.

### Pitfall 2: Duplicate Images in Different Splits

**What goes wrong:** Two identical images (same phash) end up in train and test. The model memorizes the training copy and gets a "free" correct prediction on the test copy.
**Why it happens:** Standard stratified splitting is unaware of image content similarity.
**How to avoid:** Before splitting, group exact duplicates by phash. Force all images in a duplicate group to the same split. Log which groups were found.
**Warning signs:** Suspiciously high test accuracy on specific images that happen to have duplicates in training.

### Pitfall 3: PIL Image Mode Mismatch

**What goes wrong:** Some radiographs may be grayscale (mode 'L') or have alpha channels (mode 'RGBA'). The augmentation pipeline expects 3-channel RGB (HxWx3).
**Why it happens:** Not all JPEG files are encoded identically.
**How to avoid:** Always call `image.convert("RGB")` after `Image.open()`. This converts grayscale to 3-channel and strips alpha.
**Warning signs:** RuntimeError about tensor shape mismatches in DataLoader collation.

### Pitfall 4: Center 3 Normal Representation in Center-Holdout

**What goes wrong:** Center 3 has only 27 Normal images out of 259 total. When Centers 2+3 form the test set, the Normal class is severely underrepresented (27+259=286 Normal out of 808 test images... actually: Center 2 has 259 Normal + Center 3 has 27 Normal = 286 Normal out of 808 total test images).
**Why it happens:** Data collection bias at Center 3 (MedPix favors pathological cases).
**How to avoid:** Cannot avoid -- this is inherent to the data. Document it explicitly in the split script output and reference it in the audit report. Print class distribution per split to console output.
**Warning signs:** Very different per-class metrics between stratified and center-holdout evaluations.

### Pitfall 5: Forgetting `split` Column in CSV

**What goes wrong:** Success criteria specify CSVs must contain `image_id`, `split`, and `label` columns. If `split` column is omitted, the CSVs technically fail the requirement even if they are saved as separate files.
**Why it happens:** When saving separate files per split (e.g., `stratified_train.csv`), the split is encoded in the filename, making the `split` column feel redundant.
**How to avoid:** Include the `split` column in every CSV even though it is redundant with the filename. This satisfies DATA-06 explicitly.
**Warning signs:** Verification checks fail on column presence.

### Pitfall 6: Image Path Resolution with Mixed Extensions

**What goes wrong:** Code assumes all images are `.jpeg` and cannot find the 27 `.jpg` files.
**Why it happens:** The `image_id` column in `dataset.csv` contains the actual filename with extension (e.g., `IMG000001.jpeg` or `IMG003720.jpg`). If code strips extensions and re-adds `.jpeg`, it breaks for the 27 `.jpg` files.
**How to avoid:** Use `image_id` as-is from the CSV. The filename IS the ID. Do not manipulate extensions.
**Warning signs:** FileNotFoundError for ~27 images during dataset loading.

### Pitfall 7: Rotate Border Artifacts

**What goes wrong:** When rotating images, black triangular corners appear (fill with zeros). These artifacts could confuse the model.
**Why it happens:** Albumentations `Rotate` with `border_mode=cv2.BORDER_CONSTANT` and `fill=0` fills exposed corners with black pixels.
**How to avoid:** Use `border_mode=0` (BORDER_CONSTANT) with `fill=0` (black) -- this is the standard approach for medical imaging where reflect/wrap would create misleading anatomy. The small rotation (+/-15 degrees) produces minimal border area. Alternative: use `crop_border=True` to crop out borders, but this changes the aspect ratio.
**Warning signs:** Visual inspection of augmented samples shows black triangles.

## Code Examples

### Complete Split Script Logic

```python
# Source: scikit-learn 1.7.2 train_test_split + pandas + imagehash

import pandas as pd
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

def derive_label(row: pd.Series) -> str:
    """Derive 3-class label from binary columns in dataset.csv."""
    if row["malignant"] == 1:
        return "Malignant"
    elif row["benign"] == 1:
        return "Benign"
    else:
        return "Normal"

def compute_duplicate_groups(images_dir: Path, image_ids: list[str]) -> dict[str, int]:
    """Compute phash-based duplicate groups. Returns image_id -> group_id mapping."""
    hashes = {}
    for img_id in image_ids:
        path = images_dir / img_id
        if path.exists():
            with Image.open(path) as img:
                hashes[img_id] = imagehash.phash(img, hash_size=8)

    # Group by exact hash match
    hash_groups = defaultdict(list)
    for img_id, h in hashes.items():
        hash_groups[str(h)].append(img_id)

    # Assign group IDs
    group_map = {}
    group_id = 0
    for hash_val, members in hash_groups.items():
        for member in members:
            group_map[member] = group_id
        group_id += 1

    return group_map

def save_split_csv(
    df: pd.DataFrame,
    split_name: str,
    strategy: str,
    output_dir: Path,
) -> Path:
    """Save split CSV with image_id, split, label columns."""
    out = df[["image_id", "label"]].copy()
    out["split"] = split_name
    # Reorder columns to match spec: image_id, split, label
    out = out[["image_id", "split", "label"]]

    filepath = output_dir / f"{strategy}_{split_name}.csv"
    out.to_csv(filepath, index=False)
    return filepath
```

### Complete Transforms Module

```python
# Source: albumentations 2.0.8 docs (CLAHE, Normalize, Resize, HorizontalFlip, Rotate)

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentation pipeline.

    Order matters:
    1. CLAHE (contrast enhancement on uint8 image)
    2. HorizontalFlip (random geometric)
    3. Rotate (random geometric, +/-15 degrees)
    4. Resize (to target size)
    5. Normalize (ImageNet stats, converts to float)
    6. ToTensorV2 (HWC -> CHW tensor)
    """
    return A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, fill=0, p=0.5),
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation/test preprocessing (deterministic only)."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

# Test transforms are identical to val transforms
get_test_transforms = get_val_transforms
```

### Complete Dataset Class

```python
# Source: PyTorch Dataset tutorial + albumentations pytorch-classification example

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

CLASS_TO_IDX = {"Normal": 0, "Benign": 1, "Malignant": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class BTXRDDataset(Dataset):
    """BTXRD bone tumor radiograph dataset.

    Loads images lazily from disk based on a split manifest CSV.
    Applies albumentations transforms if provided.

    Args:
        manifest_csv: Path to CSV with image_id, split, label columns.
        images_dir: Path to directory containing image files.
        transform: Albumentations Compose pipeline (or None).

    Returns:
        Tuple of (image_tensor, label_index) where:
        - image_tensor: float32 tensor of shape (3, H, W)
        - label_index: int in {0, 1, 2} for {Normal, Benign, Malignant}
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        images_dir: str | Path,
        transform=None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Validate required columns
        required_cols = {"image_id", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest CSV missing columns: {missing}")

        # Validate labels
        invalid_labels = set(self.df["label"].unique()) - set(CLASS_TO_IDX.keys())
        if invalid_labels:
            raise ValueError(f"Unknown labels in manifest: {invalid_labels}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label_str = row["label"]
        label_idx = CLASS_TO_IDX[label_str]

        # Load image with PIL (natively RGB, handles both .jpeg and .jpg)
        image_path = self.images_dir / image_id
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)  # HWC uint8 numpy array, shape (H, W, 3)

        # Apply albumentations transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]  # torch.Tensor (C, H, W)

        return image, label_idx

    @property
    def class_counts(self) -> dict[str, int]:
        """Return per-class sample counts."""
        return self.df["label"].value_counts().to_dict()

    @property
    def labels(self) -> list[int]:
        """Return all label indices (for computing class weights)."""
        return [CLASS_TO_IDX[l] for l in self.df["label"].tolist()]
```

### DataLoader Factory Function

```python
# Source: PyTorch DataLoader docs

from torch.utils.data import DataLoader

def create_dataloader(
    dataset: BTXRDDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
```

### Verification Snippet: Check Tensor Shape and Normalization

```python
# Verification code to run after implementing the dataset
def verify_dataset(dataset, expected_size=224):
    """Verify dataset returns correctly shaped and normalized tensors."""
    image, label = dataset[0]

    # Shape check
    assert image.shape == (3, expected_size, expected_size), \
        f"Expected (3, {expected_size}, {expected_size}), got {image.shape}"

    # Type check
    assert image.dtype == torch.float32, f"Expected float32, got {image.dtype}"

    # Normalization check (values should be roughly in [-2.5, 2.5] range
    # after ImageNet normalization)
    assert image.min() > -5.0, f"Min value {image.min()} seems wrong"
    assert image.max() < 5.0, f"Max value {image.max()} seems wrong"

    # Label check
    assert label in {0, 1, 2}, f"Label {label} not in expected range"

    print(f"OK: shape={image.shape}, dtype={image.dtype}, "
          f"range=[{image.min():.2f}, {image.max():.2f}], label={label}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torchvision.transforms for all augmentation | albumentations for augmentation pipeline | ~2020+ | Faster, more transforms (esp. CLAHE), numpy-based |
| `torchvision.transforms.ToTensor()` (PIL-based) | `albumentations.pytorch.ToTensorV2()` (numpy-based) | albumentations 0.5+ | Must use ToTensorV2 when using albumentations pipeline |
| Manual augmentation probability logic | `A.Compose` with per-transform `p` parameter | albumentations 0.1+ | Cleaner, composable pipeline definition |
| Single train/test split | Three-way train/val/test with held-out test | Standard practice | Prevents hyperparameter tuning on test data |
| Random split only | Stratified split required for imbalanced datasets | Standard practice | Prevents splits where minority class is missing from val/test |

**Deprecated/outdated:**
- `albumentations.pytorch.ToTensor` (v1): Renamed to `ToTensorV2` which is the current standard
- `albumentations.augmentations.transforms.CLAHE`: Import path changed; use `A.CLAHE` directly
- `torchvision.transforms.ToTensor()`: Still works but should not be mixed with albumentations pipeline

## Open Questions

1. **CLAHE Probability: Always or Random?**
   - What we know: The requirements say "CLAHE" in the augmentation pipeline. CLAHE is typically used as a preprocessing step for medical images (enhancing contrast), not as a random augmentation.
   - What's unclear: Should CLAHE be applied with `p=1.0` (always, including val/test) or `p=0.5` (randomly, train only)?
   - Recommendation: Apply CLAHE with `p=1.0` in training only. The requirement lists CLAHE alongside other training augmentations, and the success criteria specify val/test should have "only deterministic transforms (resize, normalize)." If CLAHE were desired for val/test, it would be listed there. Keep CLAHE in training pipeline only, always applied (`p=1.0`), since inconsistent contrast enhancement between train and inference would be confusing.

2. **Near-Duplicate Handling (phash distance 1-5)**
   - What we know: 20 near-duplicate pairs were found. These may be multi-angle shots of the same lesion.
   - What's unclear: Should near-duplicates also be grouped, or only exact duplicates?
   - Recommendation: Group only exact duplicates (distance=0) for the split. Near-duplicates are likely different views that legitimately represent different training/test examples. Document this decision in the split script output.

3. **Center-Holdout Val Ratio**
   - What we know: Center 1 has 2,938 images. The overall val ratio is 15% of 3,746 = 562.
   - What's unclear: Should the val set from Center 1 be sized proportionally (15% of total = 562) or as a fraction of Center 1 only (15% of 2938 = 441)?
   - Recommendation: Use 15% of Center 1's images as validation (441 images). This keeps the train/val ratio within Center 1 at 85/15, consistent with the stratified split's philosophy. Document the choice.

## Sources

### Primary (HIGH confidence)

- [scikit-learn 1.8.0 train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - API parameters, stratify parameter, random_state for reproducibility
- [scikit-learn 1.8.0 StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) - Alternative splitting approach (not recommended, but reviewed)
- [albumentations CLAHE transform](https://explore.albumentations.ai/transform/CLAHE) - clip_limit, tile_grid_size parameters, uint8 requirement
- [albumentations Normalize API](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/transforms/) - ImageNet defaults, max_pixel_value=255.0
- [albumentations Compose documentation](https://albumentations.ai/docs/2-core-concepts/pipelines/) - Pipeline composition, per-transform probability
- [albumentations Rotate transform](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/rotate/) - limit parameter, border_mode, fill
- [albumentations PyTorch classification example](https://albumentations.ai/docs/examples/pytorch-classification/) - Full Dataset class pattern with transforms
- [PyTorch Dataset tutorial](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html) - __init__, __len__, __getitem__ pattern, DataLoader usage
- [torchvision Normalize](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html) - ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Secondary (MEDIUM confidence)

- [albumentations migration guide](https://albumentations.ai/docs/examples/migrating-from-torchvision-to-albumentations/) - Transform mappings between torchvision and albumentations
- [scikit-learn GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) - Group-aware splitting (reviewed but not used for this project due to small duplicate count)

### Tertiary (LOW confidence)

- WebSearch results for best practices - confirmed by official documentation above

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are already pinned in requirements.txt, APIs verified against official docs
- Architecture: HIGH - Patterns are well-established for PyTorch image classification, verified against official tutorials and albumentations examples
- Pitfalls: HIGH - Duplicate handling verified against Phase 2 audit report (21 exact pairs), CLAHE ordering verified against albumentations docs (uint8 requirement), mixed extensions verified against dataset spec (3,719 .jpeg + 27 .jpg)

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (stable domain, no fast-moving dependencies)
