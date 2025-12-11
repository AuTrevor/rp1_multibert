# MultiBERT Scaffold

## Training

### Multiclass Classifier
To train the multiclass classifier (ScamType), run:

```bash
python ScamType/train_multiclass_classifier.py --data_file functional_test_data.csv --output_dir models/ModernBERT_multiclass
```

### Irrelevant Classifier
To train the irrelevant classifier, run:

```bash
python Irrelevant/train_irrelevant_classifier.py --data_file functional_test_data.csv --output_dir models/ModernBERT_irrelevant
```

## Reporting
HTML training reports are automatically generated after training and saved to the `training_reports` directory in the project root.
Each report includes:
- Training and Validation Loss plots
- Confusion Matrix
- Classification Report
- Training Parameters
- Date and Time of execution

Filename format: `[Model_Name]_report_YYYYMMDD_HHMMSS.html`

## Models
Trained models are saved in the `models` directory (or wherever you specify via `--output_dir`).