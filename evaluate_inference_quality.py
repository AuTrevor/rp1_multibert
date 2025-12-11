import pandas as pd
from inference import inference_pipeline, perform_data_quality_checks
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
import datetime

# Configuration
INPUT_FILE = "clean_functional_test_data.csv"
OUTPUT_FILE = "inference_quality_report.html"


def determine_expected(row):
    """
    Determines the ground truth label based on the dataset rules.
    """
    description = row.get("Description", "")

    # 1. Quality Check (matches inference logic)
    if not perform_data_quality_checks(description):
        return "Could not classify"

    # 2. Irrelevant Check
    # If irrelevant == 1, it should be "Classed as not scam"
    if row.get("irrelevant") == 1:
        return "Classed as not scam"

    # 3. Multiclass Label
    return row.get("Category")


def generate_classification_report_html(y_true, y_pred):
    """
    Generates an HTML table for the classification report.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    html = "<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>"

    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue  # Handle these separately or at bottom if desired

        html += f"""
        <tr>
            <td>{label}</td>
            <td>{metrics['precision']:.2f}</td>
            <td>{metrics['recall']:.2f}</td>
            <td>{metrics['f1-score']:.2f}</td>
            <td>{metrics['support']}</td>
        </tr>
        """
    html += "</tbody></table>"
    return html


def generate_confusion_matrix_html(y_true, y_pred, labels):
    """
    Generates a simple HTML table for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    html = '<table style="width:auto; display:inline-block;">'
    html += "<tr><th>Actual \\ Predicted</th>"
    for label in labels:
        html += f"<th>{label}</th>"
    html += "</tr>"

    for i, row_label in enumerate(labels):
        html += f"<tr><th>{row_label}</th>"
        for j, count in enumerate(cm[i]):
            # Highlight non-zero diagnols? or just plain
            style = "background-color: #e8f0fe;" if i == j else ""
            html += f'<td style="{style}">{count}</td>'
        html += "</tr>"
    html += "</table>"
    return html


def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Determining expected labels...")
    df["Expected"] = df.apply(determine_expected, axis=1)

    print("Running inference (this may take a moment)...")
    # inference_pipeline takes a DF and returns a new DF with results
    # We pass the whole DF, but strictly it only needs Description
    results_df = inference_pipeline(df)

    # Merge results back (assuming order is preserved, which it should be in inference_pipeline)
    # Ideally checking against index or description if unique.
    # The inference_pipeline returns a new DF with rows corresponding to input rows.
    # Let's concatenate side-by-side or assign columns if lengths match.

    if len(results_df) != len(df):
        print(f"Warning: Output length {len(results_df)} != Input length {len(df)}")
        # Attempt minimal join on Description if unique?
        # But Descriptions might duplicate.
        # Let's assume 1-to-1 order preservation from inference.py logic which iterates row by row.
        pass

    df["Predicted"] = results_df["Predicted Category"]
    df["Confidence"] = results_df["Confidence"]

    # Evaluation
    correct_mask = df["Expected"] == df["Predicted"]
    accuracy = accuracy_score(df["Expected"], df["Predicted"])

    # Metrics breakdown
    all_labels = sorted(
        list(set(df["Expected"].unique()) | set(df["Predicted"].unique()))
    )

    # Counts
    total = len(df)
    correct_count = correct_mask.sum()
    incorrect_count = total - correct_count

    unclassified_count = (df["Predicted"] == "Could not classify").sum()

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Generating report: {OUTPUT_FILE}...")

    # HTML Generation
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Inference Quality Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .summary-box {{ 
                background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; 
                display: flex; gap: 40px; flex-wrap: wrap;
            }}
            .metric {{ display: flex; flex-direction: column; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            
            table {{ border-collapse: collapse; width: 100%; border: 1px solid #ddd; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            
            .row-correct {{ background-color: #d4edda; }} /* Greenish */
            .row-incorrect {{ background-color: #f8d7da; }} /* Reddish */
            
            details {{ margin-top: 20px; border: 1px solid #ccc; padding: 10px; border-radius: 4px; }}
            summary {{ cursor: pointer; font-weight: bold; padding: 5px; }}
            
            .cm-container {{ overflow-x: auto; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>Inference Quality Report</h1>
        <p>Generated at: {timestamp}</p>
        
        <div class="summary-box">
            <div class="metric">
                <span class="metric-value">{accuracy:.1%}</span>
                <span class="metric-label">Accuracy</span>
            </div>
            <div class="metric">
                <span class="metric-value">{total}</span>
                <span class="metric-label">Total Samples</span>
            </div>
            <div class="metric">
                <span class="metric-value">{correct_count}</span>
                <span class="metric-label">Correct</span>
            </div>
            <div class="metric">
                <span class="metric-value">{incorrect_count}</span>
                <span class="metric-label">Incorrect</span>
            </div>
            <div class="metric">
                <span class="metric-value">{unclassified_count}</span>
                <span class="metric-label">Unable to Classify</span>
            </div>
        </div>

        <h2>Classification Report</h2>
        {generate_classification_report_html(df['Expected'], df['Predicted'])}

        <h2>Confusion Matrix</h2>
        <div class="cm-container">
            {generate_confusion_matrix_html(df['Expected'], df['Predicted'], all_labels)}
        </div>

        <details>
            <summary>Detailed Results Table ({total} rows) - Click to expand</summary>
            <table>
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th>Description</th>
                        <th>Expected</th>
                        <th>Predicted</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
    """

    for idx, row in df.iterrows():
        is_correct = row["Expected"] == row["Predicted"]
        row_class = "row-correct" if is_correct else "row-incorrect"
        conf_display = (
            f"{row['Confidence']:.4f}"
            if isinstance(row["Confidence"], (int, float))
            else str(row["Confidence"])
        )

        html_content += f"""
                    <tr class="{row_class}">
                        <td>{idx + 1}</td>
                        <td>{row['Description']}</td>
                        <td>{row['Expected']}</td>
                        <td>{row['Predicted']}</td>
                        <td>{conf_display}</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </details>
    </body>
    </html>
    """

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report saved to {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
