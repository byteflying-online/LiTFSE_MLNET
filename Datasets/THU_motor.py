"""
THU_Motor Dataset Preprocessing Class
Supported experiment modes:
  - steady: Steady-state condition diagnosis (train/val/test all from steady-state segments)
  - transient_robust: Steady-to-transient robustness evaluation (train/val from steady-state segments, test from transient segments)

Dataset description:
  - Sampling rate 12.8 kHz, 90s per file, 9 columns with no header
  - Col1: sampling time (ignored), Col2: keyphase, Col3: torque, Col4-6: vibration (3 axes), Col7-9: three-phase current
  - Uses Col4-6 (3 channels) as input
  - 9-dim multi-label classification: bearing inner, bearing outer, bearing ball, broken bar, bend, static eccentricity, dynamic eccentricity, winding short, voltage unbalance

In-file time segment (speed_circulation):
  - 0-10s   (0-128000):      Transient - acceleration
  - 10-30s  (128000-384000): Steady - target RPM
  - 30-35s  (384000-448000): Transient - deceleration
  - 35-55s  (448000-704000): Steady - half RPM
  - 55-60s  (704000-768000): Transient - acceleration
  - 60-80s  (768000-1024000): Steady - target RPM
  - 80-90s  (1024000-1152000): Transient - deceleration to stop
"""

import os
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from Datasets.get_files.generalfunciton import data_transforms1d
from Datasets.Dataset_utils.DatasetsBase import dataset
from Datasets.Dataset_utils import sequence_aug as aug
from tqdm import tqdm


def parse_multilabel(filename):
    """
    Returns a 9-length 0/1 list; healthy samples are all zeros
    Dimensions: [bearing inner, bearing outer, bearing ball, broken bar, bend, static eccentricity, dynamic eccentricity, winding short, voltage unbalance]
    """

    name = filename.replace(".csv", "")
    label = [0] * 9
    if "health" in name:
        return label
    if "inner" in name:
        label[0] = 1
    if "outer" in name:
        label[1] = 1
    if "ball" in name:
        label[2] = 1
    if "broken" in name:
        label[3] = 1
    if "bend" in name:
        label[4] = 1
    if "static_eccentricity" in name:
        label[5] = 1
    if "dynamic_eccentricity" in name:
        label[6] = 1
    if "winding" in name:
        label[7] = 1
    if "voltage" in name:
        label[8] = 1
    return label


def extract_rpm(filename):
    """Extract target RPM from filename, e.g. '_20Nm_3000rpm_' or '_20Nm_1000rpm.csv' -> 3000/1000"""
    match = re.search(r"_(\d+)rpm_", filename)
    if not match:
        match = re.search(r"_(\d+)rpm\.csv$", filename)
    return int(match.group(1)) if match else None


LABEL_NAMES = [
    "bearing inner",
    "bearing outer",
    "bearing ball",
    "broken bar",
    "bend",
    "static eccentricity",
    "dynamic eccentricity",
    "winding short",
    "voltage unbalance",
]


# Steady-state segments: (start sample index, end sample index), left-closed right-open
STEADY_SEGMENTS = [
    (128_000, 384_000),  # 10-30s, target RPM
    (448_000, 704_000),  # 35-55s, half RPM
    (768_000, 1_024_000),  # 60-80s, target RPM
]

# Transient segments: (start sample index, end sample index), left-closed right-open
TRANSIENT_SEGMENTS = [
    (0, 128_000),  # 0-10s,  acceleration
    (384_000, 448_000),  # 30-35s, deceleration
    (704_000, 768_000),  # 55-60s, acceleration
    (1_024_000, 1_152_000),  # 80-90s, deceleration to stop
]


# ──────────────────────── THU_Motor Class ────────────────────────


class THU_Motor:
    NUM_CHANNELS = 3
    SAMPLING_RATE = 12800
    inputchannel = NUM_CHANNELS
    num_classes = 9

    def __init__(self, args):
        self.args = args
        self.data_dir = args["data_dir"]
        self.data_name = args.get("data_name", "THU_Motor_speed")
        self.normlizetype = args.get("normlizetype", "mean-std")
        self.test_size = args.get("test_size", 0.2)
        self.val_size = args.get("val_size", 0.2)
        self.data_mode = args.get("data_mode", "transient_robust")
        self.max_files_per_class = args.get("max_files_per_class", None)
        # Window size and step size: read from args; datasets with different window sizes are cached to different directories
        self.window_size = args.get("data_signalsize", 8192)
        self.overlap_ratio = args.get("overlap_ratio", 0.75)
        self.step_size = int(self.window_size * (1 - self.overlap_ratio))

    def data_preprare(self, signal_size=None, SNR=None):
        """
        Main function: returns (train_dataset, val_dataset, test_dataset) and label_name
        Supports caching: loads directly if cache exists; otherwise preprocesses and saves.
        """

        cache_suffix = self.data_mode + f"_ws{self.window_size}_ol{self.overlap_ratio}"
        cache_dir = os.path.join(self.data_dir, "processed", self.data_name, cache_suffix)
        os.makedirs(cache_dir, exist_ok=True)
        cache_files = {
            "train_x": os.path.join(cache_dir, "train_x.npy"),
            "train_y": os.path.join(cache_dir, "train_y.npy"),
            "val_x": os.path.join(cache_dir, "val_x.npy"),
            "val_y": os.path.join(cache_dir, "val_y.npy"),
            "test_x": os.path.join(cache_dir, "test_x.npy"),
            "test_y": os.path.join(cache_dir, "test_y.npy"),
            "label_name": os.path.join(cache_dir, "label_name.pkl"),
        }

        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            train_x = np.load(cache_files["train_x"])
            train_y = np.load(cache_files["train_y"])
            val_x = np.load(cache_files["val_x"])
            val_y = np.load(cache_files["val_y"])
            test_x = np.load(cache_files["test_x"])
            test_y = np.load(cache_files["test_y"])
            with open(cache_files["label_name"], "rb") as f:
                label_names = pickle.load(f)
            print(f"Cache files located at {cache_dir}")
            print(f"Loading completed: train_x {train_x.shape}, val_x {val_x.shape}, test_x {test_x.shape}")

            # Sample-level label combination statistics
            from collections import Counter

            def to_binary_str(label_vec):
                return "".join(str(int(v)) for v in label_vec)

            train_counter = Counter()
            val_counter = Counter()
            test_counter = Counter()

            for y in train_y:
                train_counter[to_binary_str(y)] += 1
            for y in val_y:
                val_counter[to_binary_str(y)] += 1
            for y in test_y:
                test_counter[to_binary_str(y)] += 1

            all_keys = sorted(set(train_counter.keys()) | set(val_counter.keys()) | set(test_counter.keys()))

            print("\n========== Sample-level Label Combination Statistics ==========")
            print(
                f"{'Label Combination':12s} {'Training Samples':>12s} {'Validation Samples':>12s} {'Testing Samples':>12s} {'Total':>12s}"
            )
            print("-" * 60)
            for key in all_keys:
                train_cnt = train_counter.get(key, 0)
                val_cnt = val_counter.get(key, 0)
                test_cnt = test_counter.get(key, 0)
                total = train_cnt + val_cnt + test_cnt
                print(f"{key:12s} {train_cnt:12d} {val_cnt:12d} {test_cnt:12d} {total:12d}")

            # Sample-level fault dimension statistics
            dim_names = [
                "bearing inner",
                "bearing outer",
                "bearing ball",
                "broken bar",
                "bend",
                "static eccentricity",
                "dynamic eccentricity",
                "winding short",
                "voltage unbalance",
            ]

            def count_dim(y_array):
                counts = [0] * 9
                for y in y_array:
                    for i in range(9):
                        if y[i] > 0.5:
                            counts[i] += 1
                return counts

            train_dim = count_dim(train_y)
            val_dim = count_dim(val_y)
            test_dim = count_dim(test_y)

            print("\n========== Fault dimension statistics (each fault counted independently) ==========")
            print(
                f"{'Fault Type':20s} {'Training Samples':>12s} {'Validation Samples':>12s} {'Testing Samples':>12s} {'Total':>12s}"
            )
            print("-" * 70)
            for i, dim in enumerate(dim_names):
                t, v, te = train_dim[i], val_dim[i], test_dim[i]
                total = t + v + te
                print(f"{dim:20s} {t:12d} {v:12d} {te:12d} {total:12d}")
            print("==========================================\n")

            return self._build_datasets(train_x, train_y, val_x, val_y, test_x, test_y, label_names)

        print("Cache not found, starting preprocessing...")

        # Determine folder name
        if self.data_name == "THU_Motor_speed":
            folder_name = "MCC5-THU-Motor_speed_circulation"
        elif self.data_name == "THU_Motor_torque":
            folder_name = "MCC5-THU-Motor_torque_circulation"
        else:
            raise ValueError("data_name must be 'THU_Motor_speed' or 'THU_Motor_torque'")

        folder_path = os.path.join(self.data_dir, folder_name)

        # Get all CSV files
        all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
        print(f"Found {len(all_files)} CSV files")

        # Print all file names and their 9D labels (for verification)
        print("\n========== File Names and 9D Labels ==========")
        for f in all_files:
            label = parse_multilabel(f)
            print(f"{f:60s} -> {label}")
        print("==========================================\n")

        # ── File-level split ──
        file_labels = ["".join(str(v) for v in parse_multilabel(f)) for f in all_files]

        if self.max_files_per_class is not None:
            label_to_files = {}
            for f, lb in zip(all_files, file_labels):
                label_to_files.setdefault(lb, []).append(f)
            sampled_files = []
            sampled_labels = []
            for lb, files in label_to_files.items():
                n = min(self.max_files_per_class, len(files))
                sampled_files.extend(files[:n])
                sampled_labels.extend([lb] * n)
            all_files = sampled_files
            file_labels = sampled_labels
            print(f"Max files per class limit={self.max_files_per_class}，actually using {len(all_files)} files")

        # First-level split: train_val / test (file-level)
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            all_files, file_labels, test_size=self.test_size, random_state=42, stratify=file_labels
        )

        # Second-level split: train / val (val_size as proportion of train_val)
        train_files, val_files, _, _ = train_test_split(
            train_val_files, train_val_labels, test_size=self.val_size, random_state=42, stratify=train_val_labels
        )

        # Print split details
        # Generate file_labels for statistics (required for all modes)
        file_labels_for_print = ["".join(str(v) for v in parse_multilabel(f)) for f in all_files]
        self._print_split_details(all_files, file_labels_for_print, train_files, val_files, test_files)

        if self.data_mode == "steady":
            # train, val, test all use steady-state segments
            train_segments = STEADY_SEGMENTS
            val_segments = STEADY_SEGMENTS
            test_segments = STEADY_SEGMENTS
        elif self.data_mode == "transient_robust":
            # train, val use steady-state segments; test uses transient segments
            train_segments = STEADY_SEGMENTS
            val_segments = STEADY_SEGMENTS
            test_segments = TRANSIENT_SEGMENTS
        else:
            raise ValueError(
                f"Unknown data_mode: {self.data_mode}. Only 'steady' and 'transient_robust' are supported."
            )

        print("Extracting training set samples...")
        train_x, train_y = self._extract_windows(folder_path, train_files, train_segments, desc="Training files")
        print("Extracting validation set samples...")
        val_x, val_y = self._extract_windows(folder_path, val_files, val_segments, desc="Validation files")
        print("Extracting test set samples...")
        test_x, test_y = self._extract_windows(folder_path, test_files, test_segments, desc="Test files")

        np.save(cache_files["train_x"], train_x)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["val_x"], val_x)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["test_x"], test_x)
        np.save(cache_files["test_y"], test_y)
        with open(cache_files["label_name"], "wb") as f:
            pickle.dump(LABEL_NAMES, f)

        print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
        print(f"val_x shape: {val_x.shape}, val_y shape: {val_y.shape}")
        print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")
        print(f"test_y min: {test_y.min()}, max: {test_y.max()}")
        return self._build_datasets(train_x, train_y, val_x, val_y, test_x, test_y, LABEL_NAMES)

    def _extract_windows(self, folder_path, file_list, segments, desc="Files"):
        """
        Extract samples from specified time segments of specified files using sliding windows.
        Returns: (x_array, y_array)
          x_array: (N, C, W)  C channels × W points
          y_array: (N, 9)     9-dim multi-label
        """

        x_list = []
        y_list = []
        ws = self.window_size
        ss = self.step_size

        for filename in tqdm(file_list, desc=desc):
            filepath = os.path.join(folder_path, filename)
            raw = np.loadtxt(filepath, delimiter=",")

            # Take columns 4-6 (indices 3-5) as vibration signals
            data = raw[:, 3:6]

            label = parse_multilabel(filename)

            for seg_start, seg_end in segments:
                if data.shape[0] < seg_end:
                    print(f"Warning: {filename} data length is not enough for segment ({seg_start},{seg_end})")
                    continue
                segment_data = data[seg_start:seg_end]

                if segment_data.shape[0] < ws:
                    continue

                num_windows = (segment_data.shape[0] - ws) // ss + 1
                for i in range(num_windows):
                    start = i * ss
                    window = segment_data[start : start + ws]  # (ws, 3)
                    x_list.append(window.T)
                    y_list.append(label)

        x_array = np.array(x_list, dtype=np.float32)
        y_array = np.array(y_list, dtype=np.float32)
        return x_array, y_array

    def _normalize(self, train_x, val_x, test_x):
        """
        Channel-level normalization using training set statistics.
        Per-channel independent normalization: axis=(0,2) computes per-channel statistics
        """

        if self.normlizetype == "mean-std":
            # per-channel mean and std
            mean = train_x.mean(axis=(0, 2), keepdims=True)
            std = train_x.std(axis=(0, 2), keepdims=True) + 1e-8
            train_x = (train_x - mean) / std
            val_x = (val_x - mean) / std
            test_x = (test_x - mean) / std
        elif self.normlizetype == "0-1":
            min_val = train_x.min(axis=(0, 2), keepdims=True)
            max_val = train_x.max(axis=(0, 2), keepdims=True)
            range_val = max_val - min_val + 1e-8
            train_x = (train_x - min_val) / range_val
            val_x = (val_x - min_val) / range_val
            test_x = (test_x - min_val) / range_val
        elif self.normlizetype == "-1-1":
            min_val = train_x.min(axis=(0, 2), keepdims=True)
            max_val = train_x.max(axis=(0, 2), keepdims=True)
            range_val = max_val - min_val + 1e-8
            train_x = 2 * (train_x - min_val) / range_val - 1
            val_x = 2 * (val_x - min_val) / range_val - 1
            test_x = 2 * (test_x - min_val) / range_val - 1
        else:
            raise ValueError(f"Unknown normlizetype: {self.normlizetype}")
        return train_x, val_x, test_x

    def _print_split_details(self, all_files, file_labels, train_files, val_files, test_files):
        """Print file-level split details, including label combination statistics, per-dimension category statistics, and RPM group statistics"""
        print("\n========== File-level Split Details ==========")
        print(f"data_mode: {self.data_mode}")
        print(f"Total files: {len(all_files)}")
        print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}, Test files: {len(test_files)}")

        # ── RPM group statistics ──
        rpm_groups = {}
        for f in all_files:
            rpm = extract_rpm(f) or 0
            rpm_groups.setdefault(rpm, []).append(f)
        print(f"\nRPM group distribution:")
        for rpm in sorted(rpm_groups.keys()):
            total = len(rpm_groups[rpm])
            n_train = sum(1 for f in train_files if extract_rpm(f) == rpm)
            n_val = sum(1 for f in val_files if extract_rpm(f) == rpm)
            n_test = sum(1 for f in test_files if extract_rpm(f) == rpm)
            print(f"  {rpm}rpm: Total {total} files, Training {n_train}, Validation {n_val}, Test {n_test}")

        # ── Label combination statistics ──
        def count_labels(files):
            label_counts = {}
            for f in files:
                lb = "".join(str(v) for v in parse_multilabel(f))
                label_counts[lb] = label_counts.get(lb, 0) + 1
            return label_counts

        train_counts = count_labels(train_files)
        val_counts = count_labels(val_files)
        test_counts = count_labels(test_files)

        label_to_example = {}
        for f in all_files:
            lb = "".join(str(v) for v in parse_multilabel(f))
            if lb not in label_to_example:
                label_to_example[lb] = f

        all_labels = sorted(set(list(train_counts.keys()) + list(val_counts.keys()) + list(test_counts.keys())))

        print(
            f"\n{'Label Combination':12s} {'Example Filename':45s} {'Training':>6s} {'Validation':>6s} {'Test':>6s} {'Total':>6s}"
        )
        print("-" * 100)
        for lb in all_labels:
            t = train_counts.get(lb, 0)
            v = val_counts.get(lb, 0)
            te = test_counts.get(lb, 0)
            total = t + v + te
            example = label_to_example.get(lb, "N/A")
            example_short = example.replace(".csv", "")[:45]
            print(f"{lb:12s} {example_short:45s} {t:>6d} {v:>6d} {te:>6d} {total:>6d}")

        # ── Per-dimension fault category statistics (file-level) ──
        dim_names = [
            "bearing inner",
            "bearing outer",
            "bearing ball",
            "broken bar",
            "bend",
            "static eccentricity",
            "dynamic eccentricity",
            "winding short",
            "voltage unbalance",
        ]

        def count_per_dim(files):
            dim_counts = [0] * 9
            for f in files:
                lb = parse_multilabel(f)
                for d in range(9):
                    if lb[d] == 1:
                        dim_counts[d] += 1
            return dim_counts

        train_dim = count_per_dim(train_files)
        val_dim = count_per_dim(val_files)
        test_dim = count_per_dim(test_files)

        print(f"\n{'Fault Dimension':20s} {'Training':>6s} {'Validation':>6s} {'Test':>6s} {'Total':>6s}")
        print("-" * 60)
        for d in range(9):
            t, v, te = train_dim[d], val_dim[d], test_dim[d]
            total = t + v + te
            print(f"{dim_names[d]:20s} {t:>6d} {v:>6d} {te:>6d} {total:>6d}")
        # Healthy class (all-zero label) counted separately
        health_train = sum(1 for f in train_files if parse_multilabel(f) == [0] * 9)
        health_val = sum(1 for f in val_files if parse_multilabel(f) == [0] * 9)
        health_test = sum(1 for f in test_files if parse_multilabel(f) == [0] * 9)
        health_total = health_train + health_val + health_test
        print(f"{'health(all 0)':20s} {health_train:>6d} {health_val:>6d} {health_test:>6d} {health_total:>6d}")

        # ── Categories included in each split (label combination list) ──
        print(f"\nTraining set contains {len(train_counts)} label combinations: {sorted(train_counts.keys())}")
        print(f"Validation set contains {len(val_counts)} label combinations: {sorted(val_counts.keys())}")
        print(f"Test set contains {len(test_counts)} label combinations: {sorted(test_counts.keys())}")

        # ── Balance check ──
        train_total = len(train_files)
        val_total = len(val_files)
        test_total = len(test_files)
        print(
            f"\nFile ratios: Training={train_total/len(all_files):.2f}, Validation={val_total/len(all_files):.2f}, Test={test_total/len(all_files):.2f}"
        )
        expected_ratio_test = self.test_size
        print(f"Expected test ratio: {expected_ratio_test:.2f}")

        for d in range(9):
            total = train_dim[d] + val_dim[d] + test_dim[d]
            if total > 0:
                actual_test_ratio = test_dim[d] / total
                deviation = abs(actual_test_ratio - expected_ratio_test)
                if deviation > 0.05:
                    print(
                        f"  Warning: {dim_names[d]} test ratio={actual_test_ratio:.2f}, deviation from expected{expected_ratio_test:.2f} is {deviation:.2f}"
                    )
        # Healthy class check
        actual_health_test = health_test / health_total if health_total > 0 else 0
        deviation_h = abs(actual_health_test - expected_ratio_test)
        if deviation_h > 0.05:
            print(
                f"  Warning: health test ratio={actual_health_test:.2f}, deviation from expected{expected_ratio_test:.2f} is {deviation_h:.2f}"
            )
        if all(
            abs((train_dim[d] + val_dim[d] + test_dim[d]) / (train_dim[d] + val_dim[d] + test_dim[d] + 1e-9) - 1) < 0.1
            for d in range(9)
            if (train_dim[d] + val_dim[d] + test_dim[d]) > 0
        ):
            print("  All category proportions within splits have deviation <5%, split is well-balanced")
        else:
            print("  Warning: category split deviation >5%, please note")

        # Print specific file name allocations
        print("\n--- Training Set Files ---")
        for f in sorted(train_files):
            rpm = extract_rpm(f)
            print(f"  {f}  [{rpm}rpm]")
        print("\n--- Validation Set Files ---")
        for f in sorted(val_files):
            rpm = extract_rpm(f)
            print(f"  {f}  [{rpm}rpm]")
        print("\n--- Test Set Files ---")
        for f in sorted(test_files):
            rpm = extract_rpm(f)
            print(f"  {f}  [{rpm}rpm]")
        print("==========================================\n")

    def _build_datasets(self, train_x, train_y, val_x, val_y, test_x, test_y, label_names):
        """Convert numpy arrays to dataset objects, compatible with DataLoader"""
        train_pd = pd.DataFrame({"data": list(train_x), "label": list(train_y)})
        val_pd = pd.DataFrame({"data": list(val_x), "label": list(val_y)})
        test_pd = pd.DataFrame({"data": list(test_x), "label": list(test_y)})

        train_transform = data_transforms1d(aug, "train", self.normlizetype, aug_flag=True)
        val_transform = data_transforms1d(
            aug, "val", self.normlizetype, aug_flag=False
        )  # No augmentation for validation set
        test_transform = data_transforms1d(
            aug, "val", self.normlizetype, aug_flag=False
        )  # No augmentation for test set

        train_ds = dataset(train_pd, transform=train_transform)
        val_ds = dataset(val_pd, transform=val_transform)
        test_ds = dataset(test_pd, transform=test_transform)

        return (train_ds, val_ds, test_ds), label_names


# ──────────────────────── test ────────────────────────

if __name__ == "__main__":
    args = {
        "data_dir": r"D:\data\MCC5_THU_Motor",
        "data_name": "THU_Motor_speed",
        "normlizetype": "mean-std",
        "test_size": 0.2,
        "val_size": 0.2,
        "data_mode": "steady",
        "max_files_per_class": None,
    }
    thu = THU_Motor(args)
    (train_ds, val_ds, test_ds), label_names = thu.data_preprare()
    print("Data preparation completed")
    print(f"Training set size: {len(train_ds)}, Validation set size: {len(val_ds)}, Test set size: {len(test_ds)}")
    print(f"Label names: {label_names}")
