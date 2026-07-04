# LiTFSE-MLNet: Lightweight Frequency-Domain Multi-Label Attention Network

## Dataset Setup

Download the THU Motor dataset from:

> https://data.mendeley.com/public-api/zip/6s3dggj9mw/download/1


After downloading, ensure the following folder structure:

```
MCC5_THU_Motor/
├── MCC5-THU-Motor_speed_circulation/    (144 CSV files)
└── MCC5-THU-Motor_torque_circulation/   (144 CSV files)
```

Place the `MCC5_THU_Motor` folder at your preferred location, then specify its path via the `--data_dir` argument when running the program:



## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the preprocessed dataset (must be done before training):
```bash
python main.py --data_dir /path/to/MCC5_THU_Motor --gen_data_only True
```
This step reads raw CSV files, extracts sliding-window samples, and saves cached `.npy` files under `data_dir/processed/`. It only needs to run once per configuration.

3. Run training with default settings (steady-to-transient robustness evaluation):
```bash
python main.py --data_dir /path/to/MCC5_THU_Motor --gen_data_only False
```

4. Run transient_robust diagnosis experiment:
```bash
python main.py --data_dir /path/to/MCC5_THU_Motor --data_name THU_Motor_speed --data_mode transient_robust
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | (required) | Root directory of the MCC5_THU_Motor dataset |
| `--data_name` | `THU_Motor_speed` | Dataset subfolder: `THU_Motor_speed` or `THU_Motor_torque` |
| `--data_mode` | `transient_robust` | Experiment mode: `steady` or `transient_robust` |
| `--data_signalsize` | `8192` | Signal window length (samples) |
| `--overlap_ratio` | `0.75` | Sliding window overlap ratio |
| `--model_name` | `LiTFSE_MLNet` | Model architecture |
| `--use_freqse` | `1111` | FreqSE ablation flag (4-digit: 1=use, 0=skip for conv1-conv4) |

## Notes

- Preprocessed datasets are cached under `data_dir/processed/` and automatically reused on subsequent runs.
- The program supports 9-dim multi-label fault diagnosis with 3-channel vibration input (columns 4-6 of each CSV).
