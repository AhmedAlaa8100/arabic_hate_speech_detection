# Scripts Directory

This directory contains utility scripts for the Arabic Hate Speech Detection project.

## build_combined_dataset.py

This script builds a combined Arabic hate speech dataset from two sources:

1. **manueltonneau/arabic-hate-speech-superset** - Large but imbalanced dataset
2. **IbrahimAmin/egyptian-arabic-hate-speech** - ~4k samples

### Features

- **Label Normalization**: Converts all datasets to binary classification (1 = hate/offensive, 0 = non-hate)
- **Dataset Balancing**: Balances the superset by taking ALL hate/offensive rows and randomly sampling an equal number of non-hate rows
- **Dataset Merging**: Combines both datasets into one unified dataset
- **Stratified Splitting**: Splits the combined dataset into train/val/test sets (70/15/15) with stratification
- **CSV Export**: Saves the final datasets as CSV files in `data/combined/`

### Usage

#### Basic Usage

```bash
python scripts/build_combined_dataset.py
```

#### Advanced Usage

```bash
python scripts/build_combined_dataset.py --output-dir data/combined --train-ratio 0.7 --val-ratio 0.15
```

#### Using the Helper Script

```bash
python build_dataset.py
```

### Arguments

- `--output-dir`: Output directory for combined datasets (default: `data/combined`)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)

### Output

The script creates the following files in the output directory:

- `train.csv` - Training dataset
- `val.csv` - Validation dataset
- `test.csv` - Test dataset

Each CSV file contains two columns:

- `text`: Arabic text content
- `label`: Binary label (0 = non-hate, 1 = hate/offensive)

### Requirements

- `datasets` - For loading Hugging Face datasets
- `pandas` - For data manipulation
- `scikit-learn` - For stratified splitting
- `numpy` - For numerical operations

### Logging

The script creates a log file `dataset_build.log` with detailed information about the dataset building process.

### Example Output

```
2024-01-15 10:30:00 - INFO - Starting combined dataset building process...
2024-01-15 10:30:01 - INFO - Loading manueltonneau/arabic-hate-speech-superset dataset...
2024-01-15 10:30:05 - INFO - Loaded 50000 samples from superset
2024-01-15 10:30:05 - INFO - Original label distribution: {0: 40000, 1: 10000}
2024-01-15 10:30:05 - INFO - Binary label distribution: {0: 40000, 1: 10000}
...
2024-01-15 10:30:30 - INFO - DATASET BUILDING COMPLETED SUCCESSFULLY!
2024-01-15 10:30:30 - INFO - Total samples: 15000
2024-01-15 10:30:30 - INFO - Train: 10500 (70.0%)
2024-01-15 10:30:30 - INFO - Validation: 2250 (15.0%)
2024-01-15 10:30:30 - INFO - Test: 2250 (15.0%)
```
