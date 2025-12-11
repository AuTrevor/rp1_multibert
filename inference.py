import os

# Set environment variable to suppress tokenizers warning and avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForVision2Seq,
)
import logging
import sys
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
THRESHOLD = 0.8
QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_qwen_model() -> Tuple[AutoModelForVision2Seq, AutoTokenizer, torch.device]:
    """Loads the Qwen model and tokenizer."""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            QWEN_MODEL_ID, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        )
        # device_map="auto" usually handles moving to GPU/CPU, but for consistency with other funcs we can return device
        # If device_map is used, model.device is set.
        device = model.device
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Failed to load Qwen model from {QWEN_MODEL_ID}: {e}")
        raise e


def clarify_and_summarize(
    description: str, model: AutoModelForVision2Seq, tokenizer: AutoTokenizer, device
) -> str:
    """
    Uses Qwen to clarify and summarize the description.
    """
    prompt = (
        f"Please clarify and summarize the following description text to make it clearer for classification. "
        f"Keep the summary concise and focused on the key details relevant to identifying what is happening.\n\n"
        f"Description: {description}\n\nSummary:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=150, do_sample=True, temperature=0.7
        )

    # helper to clean up output, removing the prompt
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Simple parsing to get just the new text if model repeats prompt (it usually doesn't if instructed well, but casual LMs might)
    # The 'Summary:' part might be in the output if it continued generation.
    # Qwen-Instruct models are usually good at following.
    # For now, we'll return the text after the prompt length or simply the whole new text if distinct.
    # Let's just return the generated part.
    # tokenizer.decode excludes input tokens usually if we handle `generated_ids` slicing,
    # but `model.generate` returns input+output by default.
    new_tokens = generated_ids[0][inputs.input_ids.shape[1] :]
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return summary


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

    # Initialize Qwen variables (lazy loading)
    qwen_model = None
    qwen_tokenizer = None
    qwen_device = None

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

        # Step 4: Fallback if low confidence
        if multi_conf < THRESHOLD:
            logger.info(
                f"Low confidence ({multi_conf:.4f} < {THRESHOLD}). Initiating Qwen fallback."
            )

            # Load Qwen if not already loaded
            if qwen_model is None:
                logger.info("Loading Qwen model for fallback...")
                qwen_model, qwen_tokenizer, qwen_device = load_qwen_model()

            # Clarify and summarize
            summary = clarify_and_summarize(
                description, qwen_model, qwen_tokenizer, qwen_device
            )
            logger.info(f"Generated summary for re-classification.")

            # Step 6: Re-classify (same process as step 3 but on summary)
            new_pred, new_conf = predict_batch(
                [summary], multi_model, multi_tokenizer, multi_device, multi_id2label
            )[0]

            if new_conf < THRESHOLD:
                results.append(
                    {
                        "Description": description,
                        "Predicted Category": "Unable to classify",
                        "Confidence": new_conf,  # Reporting the new confidence even if failed
                    }
                )
            else:
                results.append(
                    {
                        "Description": description,
                        "Predicted Category": new_pred,
                        "Confidence": new_conf,
                    }
                )

        else:
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
