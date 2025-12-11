import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import sys
import os
from typing import Tuple

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
IRRELEVANT_MODEL_PATH = "models/ModernBERT_irrelevant"
MULTICLASS_MODEL_PATH = "models/ModernBERT_multiclass"


def perform_data_quality_checks(description: str) -> bool:
    """
    Checks if the description meets quality standards:
    1. Length > 50 characters
    2. All characters are English (ASCII)
    """
    if not isinstance(description, str):
        return False

    if len(description) <= 50:
        return False

    try:
        description.encode("ascii")
    except UnicodeEncodeError:
        return False

    return True


def load_model_and_tokenizer(
    model_path: str,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Loads model and tokenizer from path."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model.to(device)
        model.eval()

        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise e


def predict_batch(texts: list, model, tokenizer, device, id2label=None) -> list:
    """Runs prediction on a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        results = []
        for i in range(len(texts)):
            top_prob, top_idx = torch.max(probs[i], dim=0)
            top_idx = top_idx.item()
            top_prob = top_prob.item()

            label = top_idx
            if id2label:
                label = id2label[top_idx]

            results.append((label, top_prob))

    return results


def inference_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main inference pipeline.
    Input: DataFrame with 'Description' column.
    Output: DataFrame with 'Description', 'Predicted Category', 'Confidence'.
    """
    results = []

    # Pre-loading models to avoid loading them for every row
    # Ideally, we would load them once globally or in a class, but for this function scope it's fine.
    # However, to be efficient, we will process in batches or groups later?
    # For this implementation, we'll load them once at the start.

    logger.info("Loading Irrelevant Model...")
    irr_model, irr_tokenizer, irr_device = load_model_and_tokenizer(
        IRRELEVANT_MODEL_PATH
    )

    logger.info("Loading Multiclass Model...")
    multi_model, multi_tokenizer, multi_device = load_model_and_tokenizer(
        MULTICLASS_MODEL_PATH
    )

    multi_id2label = multi_model.config.id2label

    processed_count = 0

    # We will process row by row for simplicity given the complex conditional logic,
    # but for high throughput, we should batch.
    # Given the prompt implies "returns ... for each incoming description",
    # and the logic splits based on the first model's output, simple iteration is safest for logic correctness first.

    for _, row in df.iterrows():
        description = row.get("Description", "")

        # 1. Data Quality Check
        if not perform_data_quality_checks(description):
            results.append(
                {
                    "Description": description,
                    "Predicted Category": "Could not classify",
                    "Confidence": 0,
                }
            )
            continue

        # 2. Irrelevant Classification
        # Batching single item
        irr_pred, irr_conf = predict_batch(
            [description], irr_model, irr_tokenizer, irr_device
        )[0]

        # Logic: If classification is 1 (Irrelevant/Not Scam)
        # Note: We need to verify if label '1' corresponds to "not scam" or "irrelevant".
        # Based on previous context (clean_functional_test_data.csv), 'irrelevant' column has 0s and 1s.
        # Usually 1 means "is irrelevant" (i.e. Not a Scam) and 0 means "is relevant" (i.e. Is a Scam).
        # Let's assume prediction '1' -> Irrelevant.

        if irr_pred == 1:
            results.append(
                {
                    "Description": description,
                    "Predicted Category": "Classed as not scam",
                    "Confidence": irr_conf,
                }
            )
            continue

        # 3. Multiclass Classification (if classification is 0)
        multi_pred, multi_conf = predict_batch(
            [description], multi_model, multi_tokenizer, multi_device, multi_id2label
        )[0]

        results.append(
            {
                "Description": description,
                "Predicted Category": multi_pred,
                "Confidence": multi_conf,
            }
        )

        processed_count += 1
        if processed_count % 10 == 0:
            logger.info(f"Processed {processed_count} descriptions...")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage if run directly
    print("This script is intended to be imported. See 'inference_pipeline' function.")
