"""
Dermora AI - Dataset Processing, Stratified Group Splits, and Leakage Checks
Target Dataset: HAM10000 / ISIC 2018 Task 3
Licence: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)
Attribution: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection
             of multi-source dermatoscopic images of common pigmented skin lesions.
             Sci. Data 5, 180161 (2018). doi:10.1038/sdata.2018.161
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import numpy as np
import pandas as pd
from PIL import Image

# Standard 7-class HAM10000 / ISIC mapping
HAM10000_LABEL_MAP = {
    "nv": 0,     # Melanocytic nevi
    "mel": 1,    # Melanoma
    "bkl": 2,    # Benign keratosis-like lesions
    "bcc": 3,    # Basal cell carcinoma
    "akiec": 4,  # Actinic keratoses and intraepithelial carcinoma
    "vasc": 5,   # Vascular lesions
    "df": 6      # Dermatofibroma
}

INDEX_TO_LABEL = {v: k for k, v in HAM10000_LABEL_MAP.items()}

LABEL_DISPLAY_NAMES = {
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma"
}

def load_metadata_dataframe(metadata_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads metadata table supporting both CSV and TAB / TSV formats.
    Handles quoted values cleanly and normalizes required string fields.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    # Determine delimiter based on suffix or file content inspection
    suffix = path.suffix.lower()
    if suffix in [".tab", ".tsv"]:
        sep = "\t"
    elif suffix == ".csv":
        sep = ","
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(path, sep=sep, quotechar='"', skipinitialspace=True)

    # Normalize required string columns by stripping extra quotes and surrounding whitespace
    for col in ["lesion_id", "image_id", "dx"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip('"').str.strip("'")

    return df

def validate_dataset_integrity(
    df: pd.DataFrame,
    available_image_stems: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Validates dataset integrity:
    1. Checks presence of required columns ('lesion_id', 'image_id', 'dx').
    2. Checks for missing / null values in required columns.
    3. Checks for unknown diagnostic labels outside HAM10000_LABEL_MAP.
    4. Checks for duplicate image IDs.
    5. Verifies every metadata image_id exists in the provided image file stems.
    """
    errors: List[str] = []
    warnings: List[str] = []

    required_cols = {"lesion_id", "image_id", "dx"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Metadata missing required column(s): {sorted(list(missing_cols))}")

    # Check null / missing values in required columns
    for col in required_cols.intersection(df.columns):
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            errors.append(f"Column '{col}' contains {n_missing} missing value(s).")

    # Check unknown diagnostic labels
    unknown_labels: List[str] = []
    if "dx" in df.columns:
        unique_labels = set(df["dx"].dropna().str.lower().unique())
        unknown_labels = sorted(list(unique_labels - set(HAM10000_LABEL_MAP.keys())))
        if unknown_labels:
            errors.append(f"Encountered unrecognized diagnostic label(s): {unknown_labels}")

    # Check duplicate image IDs
    duplicate_count = 0
    if "image_id" in df.columns and "dx" in df.columns:
        dup_info = check_duplicate_images(df)
        duplicate_count = dup_info["duplicate_image_ids_count"]
        if duplicate_count > 0:
            warnings.append(f"Metadata contains {duplicate_count} duplicate image_id entries.")

    # Match metadata image_ids against files on disk
    missing_images: List[str] = []
    unindexed_images: List[str] = []
    if "image_id" in df.columns and available_image_stems is not None:
        meta_images = set(df["image_id"].dropna().unique())
        missing_images = sorted(list(meta_images - available_image_stems))
        unindexed_images = sorted(list(available_image_stems - meta_images))
        if missing_images:
            errors.append(
                f"{len(missing_images)} metadata image ID(s) do not have corresponding image files on disk."
            )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "unknown_labels": unknown_labels,
        "duplicate_image_ids_count": duplicate_count,
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:10],
        "unindexed_images_count": len(unindexed_images)
    }

def verify_dataset_directory(data_dir: Path, expected_min_images: int = 10000) -> Dict[str, Any]:
    """
    Validates presence of metadata (.tab, .tsv, .csv) and image directories.
    Performs comprehensive integrity checks and reports missing files, image counts, and licence notice.
    """
    data_dir = Path(data_dir)
    metadata_candidates = [
        data_dir / "HAM10000_metadata.tab",
        data_dir / "HAM10000_metadata.tsv",
        data_dir / "HAM10000_metadata.csv",
        data_dir / "HAM10000_metadata",
        data_dir / "metadata.tab",
        data_dir / "metadata.tsv",
        data_dir / "metadata.csv"
    ]
    metadata_path = None
    for c in metadata_candidates:
        if c.exists() and c.is_file():
            metadata_path = c
            break

    image_dirs = []
    for d in ["HAM10000_images_part_1", "HAM10000_images_part_2", "images"]:
        p = data_dir / d
        if p.is_dir():
            image_dirs.append(p)

    # Fallback: if no standard image subdirectories, check if data_dir contains images directly
    if not image_dirs and data_dir.is_dir():
        if any(data_dir.glob("*.jpg")) or any(data_dir.glob("*.png")):
            image_dirs.append(data_dir)

    # Index accessible images
    available_image_stems: Dict[str, Path] = {}
    for idir in image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_file in idir.glob(ext):
                available_image_stems[img_file.stem] = img_file

    total_found_images = len(available_image_stems)

    validation_errors: List[str] = []
    validation_warnings: List[str] = []
    metadata_rows = 0
    missing_images_count = 0
    duplicate_count = 0
    unknown_labels: List[str] = []

    if metadata_path is None:
        validation_errors.append(f"No metadata file found in '{data_dir}'. Expected HAM10000_metadata.tab or .csv.")
    else:
        try:
            df = load_metadata_dataframe(metadata_path)
            metadata_rows = len(df)
            val_result = validate_dataset_integrity(df, available_image_stems=set(available_image_stems.keys()))
            validation_errors.extend(val_result["errors"])
            validation_warnings.extend(val_result["warnings"])
            missing_images_count = val_result["missing_images_count"]
            duplicate_count = val_result["duplicate_image_ids_count"]
            unknown_labels = val_result["unknown_labels"]
        except Exception as err:
            validation_errors.append(f"Failed to load or parse metadata: {err}")

    if total_found_images < expected_min_images:
        validation_warnings.append(
            f"Found {total_found_images} images, less than expected minimum ({expected_min_images})."
        )

    is_complete = (
        (metadata_path is not None)
        and (len(validation_errors) == 0)
        and (total_found_images >= expected_min_images)
    )

    return {
        "is_complete": is_complete,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata_format": metadata_path.suffix.lower() if metadata_path else None,
        "image_directories": [str(d) for d in image_dirs],
        "total_images_found": total_found_images,
        "expected_images": 10015,
        "metadata_rows": metadata_rows,
        "missing_images_count": missing_images_count,
        "duplicate_image_ids_count": duplicate_count,
        "unknown_labels": unknown_labels,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "licence": "CC BY-NC 4.0 (Non-Commercial Research & Educational Use Only)",
        "licence_warning": "CC BY-NC 4.0 strictly restricts usage to non-commercial research and education. Commercial deployment requires independent licence verification.",
        "citation": "Tschandl et al. (2018), Scientific Data, doi:10.1038/sdata.2018.161"
    }

def check_duplicate_images(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks for duplicate image IDs or conflicting diagnostic labels for the same image.
    """
    duplicate_count = int(df["image_id"].duplicated().sum())
    has_conflicts = False
    conflicts = []
    
    if duplicate_count > 0:
        dup_groups = df[df["image_id"].duplicated(keep=False)].groupby("image_id")["dx"].nunique()
        conflicts = dup_groups[dup_groups > 1].index.tolist()
        has_conflicts = len(conflicts) > 0

    return {
        "duplicate_image_ids_count": duplicate_count,
        "conflicting_diagnosis_count": len(conflicts),
        "conflicting_image_ids": conflicts
    }

def check_group_leakage(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    CRITICAL CHECK: HAM10000 has multiple images per lesion (identified by 'lesion_id').
    Verifies that NO lesion_id appears in more than one split to prevent data leakage.
    """
    train_lesions = set(train_df["lesion_id"].dropna().unique())
    val_lesions = set(val_df["lesion_id"].dropna().unique())
    
    train_val_overlap = train_lesions.intersection(val_lesions)
    
    test_lesions = set(test_df["lesion_id"].dropna().unique()) if test_df is not None else set()
    train_test_overlap = train_lesions.intersection(test_lesions) if test_df is not None else set()
    val_test_overlap = val_lesions.intersection(test_lesions) if test_df is not None else set()

    has_leakage = (len(train_val_overlap) > 0) or (len(train_test_overlap) > 0) or (len(val_test_overlap) > 0)

    return {
        "has_leakage": has_leakage,
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "train_unique_lesions": len(train_lesions),
        "val_unique_lesions": len(val_lesions),
        "test_unique_lesions": len(test_lesions),
        "train_image_count": len(train_df),
        "val_image_count": len(val_df),
        "test_image_count": len(test_df) if test_df is not None else 0
    }

def create_stratified_group_splits(
    metadata_path: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    metadata_csv_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leakage-free splitting grouped by 'lesion_id' while stratifying
    across diagnostic class 'dx' to maintain minority class proportions.
    Supports both CSV and TAB / TSV metadata files.
    """
    target_path = metadata_path if metadata_path is not None else metadata_csv_path
    if target_path is None:
        raise ValueError("Must provide metadata file path (metadata_path or metadata_csv_path).")

    df = load_metadata_dataframe(target_path)

    # Validate dataset integrity before splitting
    val_result = validate_dataset_integrity(df)
    if not val_result["is_valid"]:
        raise ValueError(f"Metadata integrity validation failed: {val_result['errors']}")

    if val_result["duplicate_image_ids_count"] > 0:
        print(f"[Warning] Found {val_result['duplicate_image_ids_count']} duplicate image_ids in metadata. Deduplicating...")
        df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    np.random.seed(random_seed)

    # 1. Group by lesion_id to determine majority diagnosis per lesion
    lesion_summary = df.groupby("lesion_id").agg({
        "dx": "first",
        "image_id": "count"
    }).reset_index()

    train_lesion_ids = []
    val_lesion_ids = []
    test_lesion_ids = []

    # 2. Stratify lesion_ids by diagnosis class
    for dx_class, group in lesion_summary.groupby("dx"):
        shuffled_lesions = group["lesion_id"].sample(frac=1.0, random_state=random_seed).tolist()
        n = len(shuffled_lesions)
        if n >= 3:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        elif n == 2:
            n_train = 1
            n_val = 1
        else:
            n_train = 1
            n_val = 0

        train_lesion_ids.extend(shuffled_lesions[:n_train])
        val_lesion_ids.extend(shuffled_lesions[n_train:n_train + n_val])
        test_lesion_ids.extend(shuffled_lesions[n_train + n_val:])

    train_df = df[df["lesion_id"].isin(set(train_lesion_ids))].copy().reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(set(val_lesion_ids))].copy().reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(set(test_lesion_ids))].copy().reset_index(drop=True)

    # 3. Assert zero leakage
    leakage = check_group_leakage(train_df, val_df, test_df)
    if leakage["has_leakage"]:
        raise RuntimeError(f"Data leakage detected during dataset partition: {leakage}")

    return train_df, val_df, test_df

class HAM10000TorchDataset:
    """
    PyTorch Dataset wrapper for HAM10000 images.
    Imports torchvision/torch inside __init__ to avoid import errors when PyTorch is not yet installed.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dirs: List[Path],
        transform=None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Build image path lookup dictionary: image_id -> full path
        self.image_path_map: Dict[str, Path] = {}
        for d in image_dirs:
            d_path = Path(d)
            if not d_path.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_file in d_path.glob(ext):
                    self.image_path_map[img_file.stem] = img_file

        missing_count = sum(1 for img_id in self.df["image_id"] if img_id not in self.image_path_map)
        if missing_count > 0:
            print(f"[Notice] {missing_count} rows in split were not found in image folders. Filtering to existing images.")
            self.df = self.df[self.df["image_id"].isin(self.image_path_map.keys())].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"]).strip().strip('"').strip("'")
        label_code = str(row["dx"]).lower().strip().strip('"').strip("'")
        label_idx = HAM10000_LABEL_MAP[label_code]

        if image_id not in self.image_path_map:
            raise FileNotFoundError(f"Image '{image_id}' not found in provided image directories.")

        img_path = self.image_path_map[image_id]
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx
