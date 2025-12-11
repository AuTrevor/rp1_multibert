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


def train_model(
    data_file,
    taxonomy_file=os.path.join(PROJECT_ROOT, "taxonomy_mapping.csv"),
    model_name_or_path="answerdotai/ModernBERT-base",
    output_dir=os.path.join(PROJECT_ROOT, "models/ModernBERT_multiclass"),
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
    # Consultant comments removed in data cleaning step

    df = df[
        df["Description"].str.len() >= 10
    ]  # Reduced limit, 50 might be too high for small test data
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
    # Handle cases where some classes might have < 2 samples
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

    # Load Tokenizer & Model
    logger.info(f"Loading model: {model_name_or_path}...")
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
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,  # Reduced warmup for potentially small data
        num_train_epochs=5,  # Increased epochs
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()

    logger.info("Evaluating...")
    print(trainer.evaluate())

    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping explicitly as json for portability (optional, as config has it)
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label2id, f, indent=4)

    # Generate HTML Report
    reporter = HTMLTrainingReporter()
    reporter.generate_report(
        trainer,
        val_dataset,
        report_dir=os.path.join(PROJECT_ROOT, "training_reports"),
        model_name="Multiclass_Classifier",
        class_names=id2label,
        start_time=start_time,
    )


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

    args = parser.parse_args()

    train_model(args.data_file, args.taxonomy_file, args.model_name, args.output_dir)
