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
import numpy as np


combined_samples = pd.read_csv("not_scam_training_data.csv")
# CSV Format:
# ID	First Resolved On	Reference number	Category Level 3	Description	Comments	Action Comments	irrelevant


import re


# Function to detect Chinese characters
def contains_chinese(text):
    mybool = False
    try:
        mybool = bool(re.search(r"[\u4e00-\u9fff]", text))
    except TypeError:
        print(text)
    return mybool


# Apply the function to the DataFrame
combined_samples["contains_chinese"] = combined_samples["Description"].apply(
    contains_chinese
)
# Filter rows that contain Chinese characters
chinese_rows = combined_samples[combined_samples["contains_chinese"]]


def remove_consultant_comments(text: str) -> str:
    """
    Finds the earliest occurrence of any delimiter from the consultant list and truncates
    the string at that point (case sensitive)

    Args:
        text: The original string to process.

    Returns:
        The truncated string, or the original string if no delimiters are found.
    """
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

    # Find the first position of each delimiter that exists in the text
    found_indices = [text.find(d) for d in ConsultantStrings if text.find(d) != -1]

    # If no delimiters were found, return the original string
    if not found_indices:
        return text

    # Otherwise, truncate at the earliest found position
    earliest_index = min(found_indices)
    return text[:earliest_index]


combined_samples["Description"] = combined_samples["Description"].apply(
    remove_consultant_comments
)
filtered_samples = combined_samples[combined_samples["Description"].str.len() >= 50]
filtered_samples.reset_index(drop=True, inplace=True)
combined_samples = filtered_samples

texts = combined_samples["Description"]
labels = combined_samples["irrelevant"]

# train/test split
train_df, val_df = train_test_split(
    combined_samples,
    test_size=0.2,
    stratify=combined_samples["irrelevant"],
    random_state=42,
)

# tokenizer init
# pre-trained BERT model
# "./bert_tokenizer" is downloaded'bert-based-uncased' model which is english BERT lowercase
tokenizer = BertTokenizerFast.from_pretrained("./ModernBERT")


# custom pytorch dataset class to wrap our tokenized reports
# this is needed to train BERT using the dataloader
class ReportDataset(torch.utils.data.Dataset):
    def __init__(self, text, labels=None):
        """this yakes in a list of report texts, optinally labels, and tokenizes them"""
        # truncation to cut off text if to long
        # padding to make all inputs same length
        # max length set to 512, max BERT can handle
        self.encodings = tokenizer(text, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        """returns one sample (one report) from the dataset at the given index"""
        # get tokenized inputs for this one sample
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

        # if we have labels, add the label for this sample to the item dict
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        return the total number of samples in the dataset
        required by PyTorch dataloader to know the dataset size
        """
        return len(self.encodings["input_ids"])


# create datasets
train_dataset = ReportDataset(
    train_df["Description"].tolist(), train_df["irrelevant"].tolist()
)
val_dataset = ReportDataset(
    val_df["Description"].tolist(), val_df["irrelevant"].tolist()
)

# compute class weight
# handles the class imbalance problem of more relevant reports than irrelevant reports

# balanced mode calculates weights based on how often each class appears
# less common classes get higher weights to make sure the model pays attention to them
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([1, 0]),  # 1 = irrelevant, 0 = relevant
    y=train_df["irrelevant"],  # labels from training data
)

# convert the result into a pytorch tensor
# needed beacause our model will use pytorch, and will need tensor inputs
class_weights = torch.tensor(class_weights, dtype=torch.float)


# calculate evaluation metrics during testing
def compute_metrics(evaluation_prediction):
    # get raw predictions and true labels
    logits, labels = evaluation_prediction

    # convert logits to predicted class (0 or 1)
    preds = np.argmax(logits, axis=1)

    # compute precision, recall and f1 scores for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average=None,  # get metric seperately
        zero_division=0,  # avoid divison error if no prediction made
    )

    # compute overall accuracy
    acc = accuracy_score(labels, preds)

    # return all metrics in a dictionary
    return {
        "accuracy": acc,
        # metrics for class 1 irrelevant
        "precision_irrelevant": precision[1],
        "recall_irrelevant": recall[1],
        "f1_irrelevant": f1[1],
        # metrics for class 0 relevant
        "precision_relevant": precision[0],
        "recall_relevant": recall[0],
        "f1_relevant": f1[0],
    }


# custom trainer class with class weighted loss
# include class weights to handle imbalance
class CustomTrainer(Trainer):
    # custom loss function (measure how wrong the model is) that includes class weigths
    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):

        # remove labels from inputs and store seperatly
        labels = inputs.pop("labels")

        # pass remaining inputs to the model
        outputs = model(**inputs)

        # extract the raw output (logits) scores from the model
        logits = outputs.logits

        # define cross entropy loss function with class weights
        # ensures model pays more attention to underrepresnted class
        loss_function = torch.nn.CrossEntropyLoss(
            weights=class_weights.to(logits.device)
        )  # moves weights to the same place as logits

        # compute actual loss by comparing predicted logits vs true labels
        loss = loss_function(logits, labels)

        # return both loss amd the model output (if set to true)
        return (loss, outputs) if return_outputs else loss


from torch import nn
from transformers import Trainer
from typing import Optional, Union, Any


class CustomTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        reduction = "mean" if num_items_in_batch is not None else "sum"
        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device), reduction=reduction
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        return (loss, outputs) if return_outputs else loss


from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import torch

# Load a sample dataset (you can replace this with your own)
dataset = train_dataset

model = BertForSequenceClassification.from_pretrained("./ModernBERT", num_labels=2)


# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints/modernbert",
    eval_strategy="steps",
    per_device_train_batch_size=192,
    per_device_eval_batch_size=192,
    warmup_steps=100,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./training_logs/modernbert",
    save_strategy="steps",
    load_best_model_at_end=True,  # after training load best model
    metric_for_best_model="f1_irrelevant",  # use f1 score on irrelevant class to decide best model,
    log_level="warning",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# trainer = CustomTrainer(
#    model=model,
#    args=training_args,
#    train_dataset=train_dataset,
#    eval_dataset=val_dataset,
#    compute_metrics=compute_metrics #function to calcualte metrics,

# )

import logging
import sys

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
# transformers.utils.logging.set_verbosity(log_level)


# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

trainer.save_model("./models/ModernBERT_trained")
tokenizer.save_pretrained("./models/ModernBERT_trained")
