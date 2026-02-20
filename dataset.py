"""
dataset.py
----------
Dataset classes for embryo development classification.

Handles:
- Multi-focal plane loading (7 focal planes per timepoint)
- Stage annotations from CSV or folder naming convention
- Train/test splits from JSON files
- Cross-validation folds
- Preprocessing transforms (CLAHE, Canny, etc.) from preprocessing_transforms.py
- Augmentation pipelines for pretraining and fine-tuning
- Temporal sampling (for temporal SSL and GRU fine-tuning)

Returns per __getitem__:
  Pretraining : (focal_imgs, focal_imgs_aug, embryo_id, time_index)
  Fine-tuning  : (focal_imgs, label, embryo_id, time_index)
  Temporal seq : (seq_focal_imgs, seq_labels, embryo_id, seq_time_indices)
"""

import os
import re
import json
import csv
import math
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Union
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Optional import of custom preprocessing transforms
# ---------------------------------------------------------------------------
try:
    from preprocessing_transforms import get_preprocessing_transform, PreprocessingConfig
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    print("[dataset] Warning: preprocessing_transforms.py not found. "
          "Set preprocessing.method=null in config to disable.")


# =============================================================================
# Utility helpers
# =============================================================================

def load_json_patients(json_path: str) -> List[str]:
    """Load patient list from JSON file.
    Supports: {"patients": [...]} or flat list [...].
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for key in ("patients", "patient_ids", "ids", "names"):
            if key in data:
                return data[key]
    raise ValueError(f"Cannot parse patient JSON from {json_path}")


def variance_of_laplacian(img_array: np.ndarray) -> float:
    """Compute sharpness score using variance of Laplacian."""
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_best_focus_plane(focal_images: List[np.ndarray]) -> int:
    """Return index of sharpest focal plane based on variance of Laplacian."""
    scores = [variance_of_laplacian(img) for img in focal_images]
    return int(np.argmax(scores))


# =============================================================================
# Annotation parsers
# =============================================================================

def parse_annotations_csv(csv_path: str) -> Dict[str, Dict[int, int]]:
    """
    Parse stage annotations from CSV.

    Expected columns: patient, stage, start_frame, end_frame
    Returns: {patient_id: {frame_idx: stage_label, ...}, ...}
    """
    annotations = defaultdict(dict)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = row["patient"].strip()
            stage = int(row["stage"])
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            for frame in range(start, end + 1):
                annotations[patient][frame] = stage - 1  # 0-indexed
    return dict(annotations)


def parse_annotations_folder(patient_dir: Path, pattern: str) -> Dict[int, int]:
    """
    Parse stage annotations from folder names inside a patient directory.

    Pattern example: "t{start:03d}_t{end:03d}_stage{stage}"
    Returns: {frame_idx: stage_label, ...}
    """
    annotations = {}
    # Build regex from pattern
    regex_pat = (
        pattern.replace("{start:03d}", r"(?P<start>\d+)")
                .replace("{end:03d}", r"(?P<end>\d+)")
                .replace("{stage}", r"(?P<stage>\d+)")
    )
    for item in patient_dir.iterdir():
        if not item.is_dir():
            continue
        m = re.fullmatch(regex_pat, item.name)
        if m:
            start = int(m.group("start"))
            end = int(m.group("end"))
            stage = int(m.group("stage")) - 1  # 0-indexed
            for frame in range(start, end + 1):
                annotations[frame] = stage
    return annotations


# =============================================================================
# Core embryo index builder
# =============================================================================

class EmbryoIndex:
    """
    Builds an index of all (patient, frame_index) samples and their labels.

    Directory structure expected:
        <data_root>/
          embryo_dataset_F1/
            <patient_id>/
              t001.png
              t002.png
              ...
          embryo_dataset_F2/
            <patient_id>/
              ...
          ...
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        data_cfg = cfg["data"]
        self.data_root = Path(data_cfg["data_root"])
        self.focal_prefix = data_cfg["focal_prefix"]
        self.focal_indices = data_cfg["focal_indices"]
        self.num_focal = data_cfg["num_focal_planes"]
        self.image_ext = data_cfg["image_ext"]
        self.num_stages = data_cfg["num_stages"]
        self.annotation_format = data_cfg["annotation_format"]
        self.annotation_folder_pattern = data_cfg.get("annotation_folder_pattern", "")

        # Load all annotations
        if self.annotation_format == "csv":
            self._all_annotations = parse_annotations_csv(data_cfg["annotation_csv"])
        else:
            self._all_annotations = {}  # lazy-loaded per patient in build()

        # Discover focal plane directories
        self.focal_dirs: List[Path] = []
        for fi in self.focal_indices:
            fdir = self.data_root / f"{self.focal_prefix}{fi}"
            if not fdir.exists():
                raise FileNotFoundError(f"Focal dir not found: {fdir}")
            self.focal_dirs.append(fdir)

        # Reference focal dir for patient/frame discovery
        self._ref_focal_dir = self.focal_dirs[0]

    def get_patients(self) -> List[str]:
        """Return all patients found in the reference focal directory."""
        return sorted([p.name for p in self._ref_focal_dir.iterdir() if p.is_dir()])

    def build(self, patients: List[str]) -> Tuple[List[dict], Dict[str, int]]:
        """
        Build sample list for given patients.

        Returns:
            samples: list of dicts with keys:
                {patient, frame_idx, label, focal_paths}
            patient_to_id: mapping from patient name to integer ID
        """
        samples = []
        patient_to_id = {p: i for i, p in enumerate(sorted(patients))}

        for patient in patients:
            # Get frame annotations
            if self.annotation_format == "csv":
                frame_annot = self._all_annotations.get(patient, {})
            else:
                # Folder-based: parse from first focal dir
                patient_dir = self._ref_focal_dir / patient
                frame_annot = parse_annotations_folder(
                    patient_dir, self.annotation_folder_pattern
                )

            # Discover frames from reference focal plane
            ref_patient_dir = self._ref_focal_dir / patient
            if not ref_patient_dir.exists():
                print(f"[EmbryoIndex] Warning: patient dir not found: {ref_patient_dir}")
                continue

            frame_files = sorted(ref_patient_dir.glob(f"*{self.image_ext}"))
            if not frame_files:
                # Try all image extensions
                for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    frame_files = sorted(ref_patient_dir.glob(f"*{ext}"))
                    if frame_files:
                        break

            for frame_path in frame_files:
                # Extract frame index from filename (e.g., t001.png → 1)
                stem = frame_path.stem
                m = re.search(r"(\d+)", stem)
                if m is None:
                    continue
                frame_idx = int(m.group(1))

                # Get label
                label = frame_annot.get(frame_idx, -1)  # -1 = unknown
                if label < 0 or label >= self.num_stages:
                    continue  # skip unlabeled frames

                # Build paths for all focal planes
                focal_paths = []
                valid = True
                for fdir in self.focal_dirs:
                    fp = fdir / patient / frame_path.name
                    if not fp.exists():
                        valid = False
                        break
                    focal_paths.append(str(fp))

                if not valid:
                    continue

                samples.append({
                    "patient": patient,
                    "frame_idx": frame_idx,
                    "label": label,
                    "focal_paths": focal_paths,
                    "embryo_id": patient_to_id[patient],
                })

        return samples, patient_to_id


# =============================================================================
# Transform builders
# =============================================================================

def build_preprocessing_transform(cfg: dict) -> Optional[Callable]:
    """Build the preprocessing transform from config."""
    prep_cfg = cfg.get("preprocessing", {})
    method = prep_cfg.get("method", None)
    if method is None or method == "null":
        return None

    if not PREPROCESSING_AVAILABLE:
        raise ImportError(
            "preprocessing_transforms.py is required but not found. "
            "Place it in the same directory or set preprocessing.method=null."
        )

    from preprocessing_transforms import PreprocessingConfig as PC

    class _DictConfig:
        """Adapter so PreprocessingConfig works with our dict config."""
        def __init__(self, d):
            self.prep = d.get("preprocessing", {})
        def get(self, *keys, default=None):
            value = self.prep
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

    return get_preprocessing_transform(method, _DictConfig(cfg))


def build_augmentation_transform(
    aug_cfg: dict,
    image_size: Tuple[int, int],
    normalize_mean: Optional[List[float]],
    normalize_std: Optional[List[float]],
    preprocessing_transform: Optional[Callable] = None,
    is_train: bool = True,
) -> transforms.Compose:
    """Build full transform pipeline: preprocessing + augmentation + toTensor."""
    transform_list = []

    # 1. Custom preprocessing (CLAHE, Canny, etc.)
    if preprocessing_transform is not None:
        transform_list.append(preprocessing_transform)

    h, w = image_size

    if is_train:
        if aug_cfg.get("random_crop", True):
            scale = aug_cfg.get("crop_scale", [0.2, 1.0])
            transform_list.append(
                transforms.RandomResizedCrop((h, w), scale=tuple(scale))
            )
        else:
            transform_list.append(transforms.Resize((h, w)))

        if aug_cfg.get("random_hflip", True):
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_cfg.get("random_vflip", False):
            transform_list.append(transforms.RandomVerticalFlip())

        if aug_cfg.get("color_jitter", False):
            strength = aug_cfg.get("color_jitter_strength", 0.4)
            transform_list.append(transforms.ColorJitter(
                brightness=0.8 * strength,
                contrast=0.8 * strength,
                saturation=0.8 * strength,
                hue=0.2 * strength,
            ))

        if aug_cfg.get("grayscale_prob", 0) > 0:
            transform_list.append(
                transforms.RandomGrayscale(p=aug_cfg["grayscale_prob"])
            )

        if aug_cfg.get("gaussian_blur", False):
            sigma = aug_cfg.get("blur_sigma", [0.1, 2.0])
            kernel = max(3, (h // 10) | 1)  # odd kernel ~10% of image size
            transform_list.append(
                transforms.GaussianBlur(kernel_size=kernel, sigma=tuple(sigma))
            )
    else:
        transform_list.append(transforms.Resize((h, w)))

    transform_list.append(transforms.ToTensor())

    if normalize_mean is not None and normalize_std is not None:
        transform_list.append(transforms.Normalize(normalize_mean, normalize_std))

    return transforms.Compose(transform_list)


# =============================================================================
# Pretraining Dataset
# =============================================================================

class EmbryoPretrainDataset(Dataset):
    """
    Dataset for self-supervised pretraining.

    __getitem__ returns a dict:
        focal_views_1: Tensor [N_focal, C, H, W]  - first augmented view
        focal_views_2: Tensor [N_focal, C, H, W]  - second augmented view (for contrastive)
        focal_raw:     Tensor [N_focal, C, H, W]  - lightly-augmented (for MAE)
        best_focus_idx: int                         - sharpest focal plane index
        embryo_id:     int
        time_index:    int
        label:         int                          - stage label (may be -1 if unlabeled)
    """

    def __init__(
        self,
        samples: List[dict],
        cfg: dict,
        transform_strong: Optional[Callable] = None,
        transform_weak: Optional[Callable] = None,
    ):
        self.samples = samples
        self.cfg = cfg
        self.transform_strong = transform_strong
        self.transform_weak = transform_weak
        self.num_focal = cfg["data"]["num_focal_planes"]

        # For temporal SSL
        # Build per-embryo time-sorted sample lists
        self._embryo_frames: Dict[str, List[int]] = defaultdict(list)
        self._embryo_sample_map: Dict[Tuple[str, int], int] = {}
        for i, s in enumerate(samples):
            self._embryo_frames[s["patient"]].append(s["frame_idx"])
            self._embryo_sample_map[(s["patient"], s["frame_idx"])] = i
        for p in self._embryo_frames:
            self._embryo_frames[p] = sorted(set(self._embryo_frames[p]))

        self.gap_bins = cfg["pretrain"]["temporal_ssl"]["gap_bins"]

    def __len__(self):
        return len(self.samples)

    def _load_focal_images(self, focal_paths: List[str]) -> List[Image.Image]:
        imgs = []
        channels = self.cfg["data"]["image_channels"]
        for fp in focal_paths:
            img = Image.open(fp)
            if channels == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            imgs.append(img)
        return imgs

    def _apply_transform(
        self, imgs: List[Image.Image], transform: Callable
    ) -> torch.Tensor:
        """Apply transform to each focal image, stack to [N_focal, C, H, W]."""
        tensors = [transform(img) for img in imgs]
        return torch.stack(tensors, dim=0)

    def _get_best_focus_idx(self, focal_paths: List[str]) -> int:
        """Compute best-focus focal plane index."""
        arrays = []
        for fp in focal_paths:
            arr = np.array(Image.open(fp).convert("L"))
            arrays.append(arr)
        return find_best_focus_plane(arrays)

    def _get_temporal_pair(self, sample: dict) -> Tuple[Optional[dict], int]:
        """
        Sample a second timepoint from the same embryo and return it along
        with the time-gap bin label.
        """
        patient = sample["patient"]
        frames = self._embryo_frames[patient]
        if len(frames) < 2:
            return None, 0

        cur_frame = sample["frame_idx"]
        max_delta = self.cfg["pretrain"]["temporal_ssl"]["max_time_delta"]

        # Sample a future frame within max_delta
        future_frames = [
            f for f in frames
            if 1 <= (f - cur_frame) <= max_delta
        ]
        if not future_frames:
            return None, 0

        next_frame = random.choice(future_frames)
        delta = next_frame - cur_frame

        # Bin the gap
        bin_label = len(self.gap_bins)  # default: last bin
        for i, boundary in enumerate(self.gap_bins):
            if delta <= boundary:
                bin_label = i
                break

        sample_idx = self._embryo_sample_map.get((patient, next_frame))
        if sample_idx is None:
            return None, 0

        return self.samples[sample_idx], bin_label

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        focal_paths = sample["focal_paths"]

        # Load PIL images for all focal planes
        focal_imgs = self._load_focal_images(focal_paths)

        # Find best-focus plane
        best_focus_idx = self._get_best_focus_idx(focal_paths)

        # Two strong-augmented views (for contrastive)
        focal_view1 = self._apply_transform(focal_imgs, self.transform_strong)
        focal_view2 = self._apply_transform(focal_imgs, self.transform_strong)

        # Weak-augmented view (for MAE reconstruction)
        focal_raw = self._apply_transform(focal_imgs, self.transform_weak)

        # Temporal pair
        temporal_sample, gap_bin = self._get_temporal_pair(sample)
        if temporal_sample is not None:
            temp_imgs = self._load_focal_images(temporal_sample["focal_paths"])
            temporal_view = self._apply_transform(temp_imgs, self.transform_weak)
        else:
            temporal_view = focal_raw.clone()

        return {
            "focal_view1": focal_view1,          # [N_focal, C, H, W]
            "focal_view2": focal_view2,          # [N_focal, C, H, W]
            "focal_raw": focal_raw,              # [N_focal, C, H, W]
            "temporal_view": temporal_view,      # [N_focal, C, H, W]
            "best_focus_idx": best_focus_idx,    # int
            "gap_bin": gap_bin,                  # int
            "embryo_id": sample["embryo_id"],    # int
            "time_index": sample["frame_idx"],   # int
            "label": sample["label"],            # int
        }


# =============================================================================
# Fine-tuning Dataset
# =============================================================================

class EmbryoFinetuneDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    __getitem__ returns a dict:
        focal_imgs: Tensor [N_focal, C, H, W]
        label:      int (0-indexed stage)
        embryo_id:  int
        time_index: int
    """

    def __init__(
        self,
        samples: List[dict],
        cfg: dict,
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.cfg = cfg
        self.transform = transform
        self.num_focal = cfg["data"]["num_focal_planes"]

    def __len__(self):
        return len(self.samples)

    def _load_and_apply(self, focal_paths: List[str]) -> torch.Tensor:
        channels = self.cfg["data"]["image_channels"]
        tensors = []
        for fp in focal_paths:
            img = Image.open(fp)
            img = img.convert("L" if channels == 1 else "RGB")
            if self.transform:
                img = self.transform(img)
            tensors.append(img)
        return torch.stack(tensors, dim=0)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        focal_imgs = self._load_and_apply(sample["focal_paths"])
        return {
            "focal_imgs": focal_imgs,
            "label": sample["label"],
            "embryo_id": sample["embryo_id"],
            "time_index": sample["frame_idx"],
        }


# =============================================================================
# Temporal Sequence Dataset (for GRU fine-tuning)
# =============================================================================

class EmbryoSequenceDataset(Dataset):
    """
    Returns a temporal window of frames for the same embryo.

    __getitem__ returns:
        focal_seq:   Tensor [T, N_focal, C, H, W]
        labels:      Tensor [T]  (stage labels per frame)
        embryo_id:   int
        time_indices: Tensor [T]
    """

    def __init__(
        self,
        samples: List[dict],
        cfg: dict,
        transform: Optional[Callable] = None,
        window_size: int = 10,
        stride: int = 1,
    ):
        self.cfg = cfg
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.num_focal = cfg["data"]["num_focal_planes"]
        self.channels = cfg["data"]["image_channels"]

        # Group frames per embryo, sorted by time
        embryo_frames: Dict[str, List[dict]] = defaultdict(list)
        for s in samples:
            embryo_frames[s["patient"]].append(s)
        for p in embryo_frames:
            embryo_frames[p].sort(key=lambda x: x["frame_idx"])

        # Build windows
        self.windows: List[List[dict]] = []
        for patient, frames in embryo_frames.items():
            for start in range(0, len(frames) - window_size + 1, stride):
                window = frames[start: start + window_size]
                if len(window) == window_size:
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        window = self.windows[idx]

        focal_seq = []
        labels = []
        time_indices = []

        for sample in window:
            imgs = []
            for fp in sample["focal_paths"]:
                img = Image.open(fp).convert("L" if self.channels == 1 else "RGB")
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            focal_seq.append(torch.stack(imgs, dim=0))  # [N_focal, C, H, W]
            labels.append(sample["label"])
            time_indices.append(sample["frame_idx"])

        return {
            "focal_seq": torch.stack(focal_seq, dim=0),    # [T, N_focal, C, H, W]
            "labels": torch.tensor(labels, dtype=torch.long),
            "embryo_id": window[0]["embryo_id"],
            "time_indices": torch.tensor(time_indices, dtype=torch.long),
        }


# =============================================================================
# DataModule: builds everything from config
# =============================================================================

class EmbryoDataModule:
    """
    Central data manager.

    Usage:
        dm = EmbryoDataModule(cfg)
        dm.setup()
        train_loader = dm.get_pretrain_loader()
        # or
        train_loader, val_loader, test_loader = dm.get_finetune_loaders()
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.index = EmbryoIndex(cfg)
        self._train_patients: Optional[List[str]] = None
        self._test_patients: Optional[List[str]] = None
        self._cv_folds: Optional[List[Tuple[List[str], List[str]]]] = None

    def setup(self):
        """Load patient splits and build sample indices."""
        data_cfg = self.cfg["data"]

        train_patients = load_json_patients(data_cfg["train_json"])
        test_patients = load_json_patients(data_cfg["test_json"])

        # Keep only patients that exist in the data
        all_available = set(self.index.get_patients())
        self._train_patients = [p for p in train_patients if p in all_available]
        self._test_patients = [p for p in test_patients if p in all_available]

        if len(self._train_patients) == 0:
            raise ValueError("No training patients found in the data directory!")

        print(f"[DataModule] Train patients: {len(self._train_patients)}, "
              f"Test patients: {len(self._test_patients)}")

        # Build sample lists
        self._train_samples, self._train_p2id = self.index.build(self._train_patients)
        self._test_samples, self._test_p2id = self.index.build(self._test_patients)

        print(f"[DataModule] Train frames: {len(self._train_samples)}, "
              f"Test frames: {len(self._test_samples)}")

        # Build CV folds if requested
        if data_cfg.get("use_cross_validation", False):
            self._build_cv_folds(data_cfg["cv_folds"])

    def _build_cv_folds(self, n_folds: int):
        """Build cross-validation folds over training patients."""
        patients = self._train_patients.copy()
        random.shuffle(patients)
        fold_size = math.ceil(len(patients) / n_folds)
        self._cv_folds = []
        for i in range(n_folds):
            val_patients = patients[i * fold_size: (i + 1) * fold_size]
            train_patients = [p for p in patients if p not in val_patients]
            train_s, train_p2id = self.index.build(train_patients)
            val_s, val_p2id = self.index.build(val_patients)
            self._cv_folds.append((train_s, val_s))
        print(f"[DataModule] Built {n_folds} CV folds")

    # ------------------------------------------------------------------
    # Transform factories
    # ------------------------------------------------------------------
    def _get_preprocessing(self) -> Optional[Callable]:
        return build_preprocessing_transform(self.cfg)

    def _get_pretrain_transforms(self):
        prep = self._get_preprocessing()
        aug_cfg = self.cfg["pretrain"]["augmentation"]
        h, w = self.cfg["data"]["image_size"]
        mean = self.cfg["preprocessing"].get("normalize_mean")
        std = self.cfg["preprocessing"].get("normalize_std")

        transform_strong = build_augmentation_transform(
            aug_cfg, (h, w), mean, std, prep, is_train=True
        )
        transform_weak = build_augmentation_transform(
            {}, (h, w), mean, std, prep, is_train=False
        )
        return transform_strong, transform_weak

    def _get_finetune_transforms(self) -> Tuple[Callable, Callable]:
        prep = self._get_preprocessing()
        aug_cfg = self.cfg["finetune"]["augmentation"]
        h, w = self.cfg["data"]["image_size"]
        mean = self.cfg["preprocessing"].get("normalize_mean")
        std = self.cfg["preprocessing"].get("normalize_std")

        train_transform = build_augmentation_transform(
            aug_cfg, (h, w), mean, std, prep, is_train=True
        )
        val_transform = build_augmentation_transform(
            {}, (h, w), mean, std, prep, is_train=False
        )
        return train_transform, val_transform

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------
    def _make_loader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        dc = self.cfg["data"]
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=dc["num_workers"],
            pin_memory=dc["pin_memory"],
            prefetch_factor=dc.get("prefetch_factor", 2) if dc["num_workers"] > 0 else None,
            drop_last=shuffle,  # drop last incomplete batch during training
        )

    def get_pretrain_loader(self) -> DataLoader:
        """Return DataLoader for pretraining (full training set)."""
        t_strong, t_weak = self._get_pretrain_transforms()
        dataset = EmbryoPretrainDataset(
            self._train_samples, self.cfg, t_strong, t_weak
        )
        bs = self.cfg["pretrain"]["batch_size"]
        return self._make_loader(dataset, bs, shuffle=True)

    def get_finetune_loaders(
        self, fold_idx: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return (train_loader, val_loader, test_loader) for fine-tuning.

        If fold_idx is set, returns the corresponding CV fold as train/val.
        """
        t_train, t_val = self._get_finetune_transforms()
        bs = self.cfg["finetune"]["batch_size"]
        use_seq = self.cfg["model"]["classifier"].get("use_temporal_gru", False)
        win = self.cfg["finetune"].get("temporal_window", 10)

        DatasetClass = EmbryoSequenceDataset if use_seq else EmbryoFinetuneDataset

        if fold_idx is not None and self._cv_folds is not None:
            train_s, val_s = self._cv_folds[fold_idx]
        else:
            # Simple train/val split: 90/10 from training patients
            n_val = max(1, int(0.1 * len(self._train_patients)))
            val_patients = set(self._train_patients[-n_val:])
            train_patients = [p for p in self._train_patients if p not in val_patients]
            train_s, _ = self.index.build(train_patients)
            val_s, _ = self.index.build(list(val_patients))

        if use_seq:
            train_ds = EmbryoSequenceDataset(train_s, self.cfg, t_train, win)
            val_ds = EmbryoSequenceDataset(val_s, self.cfg, t_val, win)
            test_ds = EmbryoSequenceDataset(self._test_samples, self.cfg, t_val, win)
        else:
            train_ds = EmbryoFinetuneDataset(train_s, self.cfg, t_train)
            val_ds = EmbryoFinetuneDataset(val_s, self.cfg, t_val)
            test_ds = EmbryoFinetuneDataset(self._test_samples, self.cfg, t_val)

        return (
            self._make_loader(train_ds, bs, shuffle=True),
            self._make_loader(val_ds, bs, shuffle=False),
            self._make_loader(test_ds, bs, shuffle=False),
        )

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for the training set."""
        num_stages = self.cfg["data"]["num_stages"]
        counts = torch.zeros(num_stages)
        for s in self._train_samples:
            if 0 <= s["label"] < num_stages:
                counts[s["label"]] += 1
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_stages
        return weights
