import os
import time
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix, classification_report
import torch

logger = logging.getLogger(__name__)


class HTMLTrainingReporter:
    def __init__(self):
        pass

    def _plot_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str

    def generate_report(
        self,
        best_trainer,
        best_val_dataset,
        report_dir,
        model_name,
        class_names=None,
        start_time=None,
        run_history=None,
    ):
        logger.info("Generating HTML training report...")
        os.makedirs(report_dir, exist_ok=True)

        # Calculate total training time for the best run
        training_time = "N/A"
        if start_time:
            elapsed = time.time() - start_time
            training_time = f"{elapsed:.2f} seconds"

        # --- Generate Comparison Table (if run_history provided) ---
        comparison_html = ""
        if run_history:
            comparison_df = pd.DataFrame(run_history)
            # Reorder columns to put metrics last
            cols = [
                c
                for c in comparison_df.columns
                if c
                not in [
                    "accuracy",
                    "f1_macro",
                    "f1_weighted",
                    "precision_macro",
                    "recall_macro",
                ]
            ]
            metrics = [
                c
                for c in comparison_df.columns
                if c
                in [
                    "accuracy",
                    "f1_macro",
                    "f1_weighted",
                    "precision_macro",
                    "recall_macro",
                ]
            ]
            comparison_df = comparison_df[cols + metrics]

            # Highlight max values in metrics
            comparison_html = comparison_df.to_html(
                classes="table table-striped",
                index=False,
                float_format="{:0.4f}".format,
            )

        # --- Best Model Detailed Analysis ---

        # Collect Metrics for the best model from its history
        log_history = best_trainer.state.log_history

        # Separate Train and Eval metrics
        train_loss = []
        eval_loss = []
        train_steps = []
        eval_steps = []

        for entry in log_history:
            if "loss" in entry and "step" in entry:
                train_loss.append(entry["loss"])
                train_steps.append(entry["step"])
            if "eval_loss" in entry and "step" in entry:
                eval_loss.append(entry["eval_loss"])
                eval_steps.append(entry["step"])

        # Create Loss Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_steps, train_loss, label="Training Loss")
        ax.plot(eval_steps, eval_loss, label="Validation Loss", linestyle="--")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Best Model Training Dynamics")
        ax.legend()
        loss_plot_b64 = self._plot_to_base64(fig)

        # Evaluate on Validation Set for Confusion Matrix
        logger.info("Running evaluation for confusion matrix on best model...")
        predictions, labels, _ = best_trainer.predict(best_val_dataset)
        preds = np.argmax(predictions, axis=1)

        # Handle class names
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
        elif isinstance(class_names, dict):
            # Ensure it's sorted by ID
            class_names = [class_names[i] for i in sorted(class_names.keys())]

        # Confusion Matrix
        cm = confusion_matrix(labels, preds)

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(cax)

        # Set ticks
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        # Add labels
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title("Confusion Matrix (Best Model)")
        cm_plot_b64 = self._plot_to_base64(fig)

        # Classification Report
        clf_report = classification_report(
            labels, preds, target_names=class_names, output_dict=True
        )
        clf_report_df = pd.DataFrame(clf_report).transpose()
        clf_report_html = clf_report_df.to_html(
            classes="table", float_format="{:0.3f}".format
        )

        # Parameters Table (Best Model)
        args_dict = best_trainer.args.to_dict()
        important_args = {
            k: args_dict[k]
            for k in [
                "learning_rate",
                "per_device_train_batch_size",
                "num_train_epochs",
                "weight_decay",
                "output_dir",
            ]
        }
        params_df = pd.DataFrame([important_args])
        params_html = params_df.to_html(classes="table", index=False)

        # Generate HTML
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

        comparison_section = ""
        if run_history:
            comparison_section = f"""
            <h2>Hyperparameter Tuning Comparison</h2>
            <p>The following table shows the results of all configurations tested:</p>
            <div style="overflow-x:auto;">
                {comparison_html}
            </div>
            <br><hr><br>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Report - {model_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }}
                .container {{ max-width: 1200px; margin: auto; background-color: #fff; padding: 40px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                .table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 12px 15px; }}
                .table th {{ background-color: #f2f2f2; text-align: left; font-weight: 600; color: #555; }}
                .table-striped tr:nth-child(even) {{ background-color: #f8f8f8; }}
                .table tr:hover {{ background-color: #f1f1f1; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 20px; border-radius: 4px; padding: 5px; }}
                .meta-info {{ background-color: #eef2f3; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #3498db; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Training Report: {model_name}</h1>
                
                <div class="meta-info">
                    <p><strong>Date:</strong> {current_time_str}</p>
                    <p><strong>Training Time (Best Model):</strong> {training_time}</p>
                </div>
                
                {comparison_section}

                <h2>Best Model Analysis</h2>
                <h3>Selected Parameters</h3>
                {params_html}

                <h3>Training Dynamics</h3>
                <img src="data:image/png;base64,{loss_plot_b64}" alt="Loss Plot">

                <h3>Performance on Validation Set</h3>
                
                <h4>Confusion Matrix</h4>
                <img src="data:image/png;base64,{cm_plot_b64}" alt="Confusion Matrix">
                
                <h4>Detailed Classification Report</h4>
                {clf_report_html}
            </div>
        </body>
        </html>
        """

        timestamp_filename = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_report_{timestamp_filename}.html"
        output_path = os.path.join(report_dir, filename)

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Report saved to {output_path}")
