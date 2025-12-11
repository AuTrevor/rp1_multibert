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


def train_single_run(
    data_file,
    model_name_or_path,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights_tensor,
):
    """
    Executes a single training run with specific hyperparameters.
    """
    run_name = f"lr{learning_rate}_bs{batch_size}_ep{epochs}"
    run_output_dir = os.path.join(output_dir, run_name)

    # Load Tokenizer & Model (Re-load for each run to reset weights)
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        )
    except OSError:
        logger.warning(
            f"Model '{model_name_or_path}' not found locally. Attempting to download or using default..."
        )
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
        output_dir=run_output_dir,
        eval_strategy="epoch",  # Evaluate every epoch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50 if epochs < 5 else 100,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{run_output_dir}/logs",
        save_strategy="no",  # Don't save every checkpoint to save space
        load_best_model_at_end=False,  # We will manually track best across runs
        logging_steps=10,
    )

    trainer = CustomTrainer(
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
    model_name_or_path="answerdotai/ModernBERT-base",
    output_dir=os.path.join(PROJECT_ROOT, "models/ModernBERT_trained"),
    epochs=None,
    batch_size=None,
    learning_rate=None,
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
    df = df[df["Description"].str.len() >= 50]
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Using {len(df)} samples after filtering.")

    X = df["Description"].tolist()
    y = df["irrelevant"].tolist()

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Compute class weights
    logger.info("Computing class weights...")
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Class weights: {class_weights}")

    # --- Hyperparameter Grid ---
    if epochs:
        epoch_options = [epochs]
    else:
        epoch_options = [3]  # Default grid

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
                    data_file,
                    model_name_or_path,
                    output_dir,
                    ep,
                    bs,
                    lr,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    class_weights_tensor,
                )

                # Extract metric of interest (F1 for class 1 - irrelevant)
                # Note: compute_metrics returns 'f1_1'
                current_f1 = eval_metrics.get("eval_f1_1", 0.0)

                run_info = {
                    "learning_rate": lr,
                    "batch_size": bs,
                    "epochs": ep,
                    "accuracy": eval_metrics.get("eval_accuracy"),
                    "f1_target": current_f1,
                    # Add other metrics for the table
                    "precision_target": eval_metrics.get("eval_precision_1"),
                    "recall_target": eval_metrics.get("eval_recall_1"),
                }
                results.append(run_info)

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_run_details = run_info
                    # We need to save this trainer/model as the current best
                    # Since the trainer is returned, we can reference it.
                    # HOWEVER, careful with memory. We might just want to save the model to disk now.
                    best_trainer = trainer  # Keep reference to last best trainer
                    best_val_dataset = val_dataset

                    logger.info(f"New Best Model Found! F1: {current_f1}")

                    # Save immediately to ensure we have it
                    logger.info(f"Saving best model to {output_dir}...")
                    trainer.save_model(output_dir)
                    tokenizer.save_pretrained(output_dir)
                else:
                    # Clear memory if not best
                    pass

    logger.info(f"Grid Search Complete. Best F1: {best_f1}")
    logger.info(f"Best Parameters: {best_run_details}")

    # Generate HTML Report
    reporter = HTMLTrainingReporter()
    if best_trainer:
        reporter.generate_report(
            best_trainer,
            best_val_dataset,
            report_dir=os.path.join(PROJECT_ROOT, "training_reports"),
            model_name="Irrelevant_Classifier",
            class_names=["Relevant", "Irrelevant"],
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
        default=os.path.join(PROJECT_ROOT, "models/ModernBERT_trained"),
        help="Output directory for saved model",
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
        args.model_name,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
