import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
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
        labels, preds, average=None, zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision_0": precision[0],
        "recall_0": recall[0],
        "f1_0": f1[0],
        "precision_1": precision[1] if len(precision) > 1 else 0,
        "recall_1": recall[1] if len(recall) > 1 else 0,
        "f1_1": f1[1] if len(f1) > 1 else 0,
    }


def train_model(
    data_file,
    model_name_or_path="answerdotai/ModernBERT-base",
    output_dir=os.path.join(PROJECT_ROOT, "models/ModernBERT_trained"),
):
    logger.info(f"Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        logger.error(f"File {data_file} not found.")
        sys.exit(1)

    # Preprocessing
    if "Description" not in df.columns or "irrelevant" not in df.columns:
        logger.error("CSV must contain 'Description' and 'irrelevant' columns.")
        sys.exit(1)

    logger.info("Preprocessing data...")
    # Consultant comments removed in data cleaning step

    df = df[df["Description"].str.len() >= 50]
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Using {len(df)} samples after filtering.")

    X = df["Description"].tolist()
    y = df["irrelevant"].tolist()

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Compute class weights for imbalanced dataset
    logger.info("Computing class weights...")
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Class weights: {class_weights}")

    # Load Tokenizer & Model
    logger.info(f"Loading model: {model_name_or_path}...")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        )
    except OSError:
        logger.warning(
            f"Model '{model_name_or_path}' not found locally. Attempting to download or using default..."
        )
        # Fallback logic could go here, but transformers usually handles this.
        raise

    train_dataset = ReportDataset(X_train, y_train, tokenizer)
    val_dataset = ReportDataset(X_val, y_val, tokenizer)

    # Define Custom Trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # Use computed class weights
            loss_fct = nn.CrossEntropyLoss(
                weight=class_weights_tensor.to(logits.device)
            )

            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1_1",  # Assuming checking F1 score of class 1 (irrelevant)
        logging_steps=10,
    )

    trainer = CustomTrainer(
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

    # Generate HTML Report
    reporter = HTMLTrainingReporter()
    reporter.generate_report(
        trainer,
        val_dataset,
        report_dir=os.path.join(PROJECT_ROOT, "training_reports"),
        model_name="Irrelevant_Classifier",
        class_names=["Relevant", "Irrelevant"],
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
        default=os.path.join(PROJECT_ROOT, "models/ModernBERT_trained"),
        help="Output directory for saved model",
    )

    args = parser.parse_args()

    train_model(args.data_file, args.model_name, args.output_dir)
