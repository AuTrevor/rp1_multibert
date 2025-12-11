import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import logging
import sys
from tqdm import tqdm
import os

# Determine the project root (one level up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def predict(input_file, model_path, output_file, text_column="Description"):
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found.")
        sys.exit(1)

    if not os.path.exists(model_path):
        logger.warning(
            f"Model path {model_path} does not exist. Please check if the model is trained."
        )

    # Load data
    logger.info(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    if text_column not in df.columns:
        logger.error(
            f"Column '{text_column}' not found in input file. available columns: {df.columns.tolist()}"
        )
        sys.exit(1)

    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Determine device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Get ID 2 Label map
    id2label = model.config.id2label
    if not id2label:
        logger.warning(
            "No id2label found in model config. Predictions will be integers."
        )

    texts = df[text_column].astype(str).tolist()
    predictions = []

    batch_size = 16

    logger.info(f"Running predictions on {len(texts)} samples...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
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
            preds = torch.argmax(probs, dim=1)

            predictions.extend(preds.cpu().tolist())

    # Map predictions to labels
    if id2label:
        # id2label keys might be strings in json, convert to int for lookup if needed
        # usually they are ints in the config object if loaded via transformers
        predicted_labels = [id2label[p] for p in predictions]
        df["Predicted_Category"] = predicted_labels
    else:
        df["Predicted_Category_ID"] = predictions

    logger.info(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "--model_path",
        default=os.path.join(PROJECT_ROOT, "models/ModernBERT_multiclass"),
        help="Path to trained model or model name",
    )
    parser.add_argument(
        "--output_file", default="multiclass_results.csv", help="Path to save results"
    )
    parser.add_argument(
        "--text_column",
        default="Description",
        help="Name of the column containing text",
    )

    args = parser.parse_args()

    predict(args.input_file, args.model_path, args.output_file, args.text_column)
