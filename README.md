# MultiBERT Scaffold

## Training

Both the Irrelevant and Multiclass classifier scripts perform an **automatic hyperparameter grid search** (iterating over Learning Rate, Batch Size, and Epochs) by default to identify the best-performing model configuration.

### Single Training Run
You can override the grid search and perform a single run by providing specific values for hyperparameters:
- `--epochs`: Number of training epochs (e.g., `3`).
- `--batch_size`: Batch size (e.g., `16`).
- `--learning_rate`: learning rate (e.g., `5e-5`).

If ANY of these are provided, the script will run *only* that configuration (using defaults for any non-specified params if partial).

### Multiclass Classifier
To train the multiclass classifier (ScamType), run:

```bash
python ScamType/train_multiclass_classifier.py --data_file functional_test_data.csv --output_dir models/ModernBERT_multiclass
```

**Standard Arguments:**
- `--data_file`: Path to the training CSV file (default: `functional_test_data.csv`).
- `--output_dir`: Directory to save the trained model (default: `models/ModernBERT_multiclass`).
- `--model_name`: Base model to use (default: `answerdotai/ModernBERT-base`).
- `--taxonomy_file`: Path to the taxonomy mapping CSV (default: `taxonomy_mapping.csv`).

**Single Run Arguments:**
- `--epochs`, `--batch_size`, `--learning_rate`

**Note:** This script filters the data to include only records where `irrelevant == 0` and the `Description` length is at least 10 characters.

### Irrelevant Classifier
To train the irrelevant classifier, run:

```bash
python Irrelevant/train_irrelevant_classifier.py --data_file functional_test_data.csv --output_dir models/ModernBERT_irrelevant
```

**Standard Arguments:**
- `--data_file`: Path to the training CSV file (default: `functional_test_data.csv`).
- `--output_dir`: Directory to save the trained model (default: `models/ModernBERT_trained`).
- `--model_name`: Base model to use (default: `answerdotai/ModernBERT-base`).

**Single Run Arguments:**
- `--epochs`, `--batch_size`, `--learning_rate`

**Note:** This script filters the data to include only records where the `Description` length is at least 50 characters.

## Reporting
HTML training reports are automatically generated after training and saved to the `training_reports` directory in the project root.

Each report includes:
- **Hyperparameter Tuning Comparison**: A table comparing all grid search runs.
- **Best Model Analysis**:
    - Training and Validation Loss plots.
    - Confusion Matrix (evaluated on the validation set).
    - Detailed Classification Report.
    - Selected Training Parameters.

The "Best Model" is selected based on the **F1 Score** (Weighted F1 for Multiclass, Class 1 F1 for Irrelevant).

Filename format: `[Model_Name]_report_YYYYMMDD_HHMMSS.html`

## Models
Trained models are saved in the `models` directory (or wherever you specify via `--output_dir`).