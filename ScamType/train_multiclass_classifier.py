import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch import nn
import numpy as np
import logging
import sys
import argparse
import os
import json
import time

# Determine the project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.training_reporter import HTMLTrainingReporter

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


class ReportDataset(torch.utils.data.Dataset):
    def __init__(self, text, labels=None, tokenizer=None):
        self.encodings = tokenizer(text, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(evaluation_prediction):
    logits, labels = evaluation_prediction
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_single_run(
    model_name_or_path,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    X_train,
    y_train,
    X_val,
    y_val,
    unique_labels,
    id2label,
    label2id,
):
    """
    Executes a single training run for multiclass classifier.
    """
    run_name = f"lr{learning_rate}_bs{batch_size}_ep{epochs}"
    run_output_dir = os.path.join(output_dir, run_name)

    # Load Tokenizer & Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    train_dataset = ReportDataset(X_train, y_train, tokenizer)
    val_dataset = ReportDataset(X_val, y_val, tokenizer)

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{run_output_dir}/logs",
        save_strategy="no",  # Save space
        load_best_model_at_end=False,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Starting training run: {run_name}")
    trainer.train()

    logger.info("Evaluating run...")
    eval_result = trainer.evaluate()

    # cleanup
    del model
    torch.cuda.empty_cache()

    return trainer, eval_result, tokenizer, val_dataset


def train_model(
    data_file,
    taxonomy_file=os.path.join(PROJECT_ROOT, "taxonomy_mapping.csv"),
    model_name_or_path="answerdotai/ModernBERT-base",
    output_dir=os.path.join(PROJECT_ROOT, "models/ModernBERT_multiclass"),
    epochs=None,
    batch_size=None,
    learning_rate=None,
):
    # 1. Load Taxonomy to build Label Mapping
    logger.info(f"Loading taxonomy from {taxonomy_file}...")
    if not os.path.exists(taxonomy_file):
        logger.error(f"Taxonomy file {taxonomy_file} not found.")
        sys.exit(1)

    df_taxonomy = pd.read_csv(taxonomy_file)
    if "New_Scam_Category" not in df_taxonomy.columns:
        logger.error(f"Column 'New_Scam_Category' not found in {taxonomy_file}")
        sys.exit(1)

    unique_labels = sorted(df_taxonomy["New_Scam_Category"].dropna().unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    logger.info(f"Found {len(unique_labels)} unique categories.")

    # 2. Load Data
    logger.info(f"Loading training data from {data_file}...")
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found.")
        sys.exit(1)

    df = pd.read_csv(data_file)

    # Preprocessing
    required_cols = ["Description", "irrelevant", "Category"]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"CSV must contain columns: {required_cols}")
        sys.exit(1)

    # Filter for relevant reports only
    logger.info("Filtering for relevant reports (irrelevant == 0)...")
    original_len = len(df)
    df = df[df["irrelevant"] == 0].copy()
    logger.info(f"Filtered {original_len} -> {len(df)} rows.")

    logger.info("Preprocessing text...")
    df = df[df["Description"].str.len() >= 10]
    df.reset_index(drop=True, inplace=True)

    # Map Labels
    df = df[df["Category"].isin(unique_labels)]  # Ensure valid labels
    df["label"] = df["Category"].map(label2id)

    if len(df) == 0:
        logger.error("No data left after preprocessing and filtering. Exiting.")
        sys.exit(1)

    X = df["Description"].tolist()
    y = df["label"].tolist()

    # Stratified split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        logger.warning(
            "Could not stratify split (possibly singleton classes). Performing random split."
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # --- Hyperparameter Grid ---
    if epochs:
        epoch_options = [epochs]
    else:
        epoch_options = [5]

    if batch_size:
        batch_sizes = [batch_size]
    else:
        batch_sizes = [8, 16]

    if learning_rate:
        learning_rates = [learning_rate]
    else:
        learning_rates = [2e-5, 5e-5]

    results = []
    best_f1 = -1.0
    best_run_details = None
    best_trainer = None
    best_val_dataset = None

    start_total_time = time.time()

    for lr in learning_rates:
        for bs in batch_sizes:
            for ep in epoch_options:
                logger.info(f"--- Running Grid: LR={lr}, BS={bs}, Epochs={ep} ---")

                trainer, eval_metrics, tokenizer, val_dataset = train_single_run(
                    model_name_or_path,
                    output_dir,
                    ep,
                    bs,
                    lr,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    unique_labels,
                    id2label,
                    label2id,
                )

                # Extract metric: Weighted F1
                current_f1 = eval_metrics.get("eval_f1", 0.0)

                run_info = {
                    "learning_rate": lr,
                    "batch_size": bs,
                    "epochs": ep,
                    "accuracy": eval_metrics.get("eval_accuracy"),
                    "f1_weighted": current_f1,
                    "loss": eval_metrics.get("eval_loss"),
                }
                results.append(run_info)

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_run_details = run_info
                    best_trainer = trainer
                    best_val_dataset = val_dataset

                    logger.info(f"New Best Model Found! Weighted F1: {current_f1}")

                    # Save immediately
                    logger.info(f"Saving best model to {output_dir}...")
                    trainer.save_model(output_dir)
                    tokenizer.save_pretrained(output_dir)
                else:
                    pass

    logger.info(f"Grid Search Complete. Best Weighted F1: {best_f1}")
    logger.info(f"Best Parameters: {best_run_details}")

    # Save label mapping explicitly as json for portability
    # We do this at the end, ensuring it matches the model we saved
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label2id, f, indent=4)

    # Generate HTML Report
    reporter = HTMLTrainingReporter()
    if best_trainer:
        reporter.generate_report(
            best_trainer,
            best_val_dataset,
            report_dir=os.path.join(PROJECT_ROOT, "training_reports"),
            model_name="Multiclass_Classifier",
            class_names=id2label,
            start_time=start_total_time,
            run_history=results,
        )
    else:
        logger.error("No model was trained successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default=os.path.join(PROJECT_ROOT, "functional_test_data.csv"),
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--model_name", default="answerdotai/ModernBERT-base", help="Model name or path"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "models/ModernBERT_multiclass"),
        help="Output directory for saved model",
    )
    parser.add_argument(
        "--taxonomy_file",
        default=os.path.join(PROJECT_ROOT, "taxonomy_mapping.csv"),
        help="Path to taxonomy mapping file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides grid)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (overrides grid)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides grid)",
    )

    args = parser.parse_args()

    train_model(
        args.data_file,
        args.taxonomy_file,
        args.model_name,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
