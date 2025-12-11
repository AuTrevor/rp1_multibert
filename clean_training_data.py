import pandas as pd
import argparse
import sys
import logging
import os

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def remove_consultant_comments(text: str) -> str:
    """
    Finds the earliest occurrence of any delimiter from the consultant list and truncates
    the string at that point (case sensitive).
    """
    if not isinstance(text, str):
        return str(text)

    ConsultantStrings = [
        "AMEAN",
        "PJORN",
        "VYUEN",
        "CMART",
        "CCOLV",
        "DMCKA",
        "KGUNT",
        "ADELE",
        "JRICH",
        "RONEI",
        "RHALL",
        "MCARR",
        "RHALL",
    ]

    found_indices = [text.find(d) for d in ConsultantStrings if text.find(d) != -1]

    if not found_indices:
        return text

    earliest_index = min(found_indices)
    return text[:earliest_index]


def is_english_only(text: str) -> bool:
    """
    Checks if the text contains only ASCII characters.
    """
    if not isinstance(text, str):
        return False
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def clean_data(input_file, output_file):
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found.")
        sys.exit(1)

    logger.info(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    if "Description" not in df.columns:
        logger.error("Column 'Description' not found in input file.")
        sys.exit(1)

    initial_count = len(df)
    logger.info(f"Initial row count: {initial_count}")

    # Remove consultant comments
    logger.info("Removing consultant comments...")
    df["Description"] = df["Description"].apply(remove_consultant_comments)

    # Filter by length
    logger.info("Filtering descriptions shorter than 50 characters...")
    df = df[df["Description"].str.len() >= 50]
    count_after_len = len(df)
    logger.info(
        f"Rows remaining after length filter: {count_after_len} (Dropped {initial_count - count_after_len})"
    )

    # Filter non-English
    logger.info("Filtering rows with non-English characters in Description...")
    df = df[df["Description"].apply(is_english_only)]
    count_after_english = len(df)
    logger.info(
        f"Rows remaining after English filter: {count_after_english} (Dropped {count_after_len - count_after_english})"
    )

    # Save to output
    logger.info(f"Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean training data CSV.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output cleaned CSV file")

    args = parser.parse_args()

    clean_data(args.input_file, args.output_file)
