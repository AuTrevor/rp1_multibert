import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import torch
import sys
import os

# Add parent directory to path to import inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference


class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_device = "cpu"

        # Patch load_model_and_tokenizer
        self.patcher_load = patch("inference.load_model_and_tokenizer")
        self.mock_load = self.patcher_load.start()
        self.mock_load.return_value = (
            self.mock_model,
            self.mock_tokenizer,
            self.mock_device,
        )

        # Patch predict_batch
        self.patcher_predict = patch("inference.predict_batch")
        self.mock_predict = self.patcher_predict.start()

        # Patch load_qwen_model
        self.patcher_load_qwen = patch("inference.load_qwen_model")
        self.mock_load_qwen = self.patcher_load_qwen.start()
        self.qwen_model = MagicMock()
        self.qwen_tokenizer = MagicMock()
        self.mock_load_qwen.return_value = (
            self.qwen_model,
            self.qwen_tokenizer,
            self.mock_device,
        )

        # Patch clarify_and_summarize
        self.patcher_clarify = patch("inference.clarify_and_summarize")
        self.mock_clarify = self.patcher_clarify.start()

    def tearDown(self):
        self.patcher_load.stop()
        self.patcher_predict.stop()
        self.patcher_load_qwen.stop()
        self.patcher_clarify.stop()

    def test_high_confidence(self):
        """Test case where initial classification has high confidence."""
        description = "This is a very clear scam attempt that is definitely longer than fifty characters to pass the check."
        df = pd.DataFrame({"Description": [description]})

        # Mock predictions:
        # 1. Irrelevant model -> 0 (Relevant)
        # 2. Multiclass model -> "Tech Support", 0.95 (High conf)
        self.mock_predict.side_effect = [
            [(0, 0.99)],  # Irrelevant
            [("Tech Support", 0.95)],  # Multiclass
        ]

        result_df = inference.inference_pipeline(df)

        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["Predicted Category"], "Tech Support")
        self.assertEqual(result_df.iloc[0]["Confidence"], 0.95)

        # Verify Qwen was NOT loaded
        self.mock_load_qwen.assert_not_called()
        self.mock_clarify.assert_not_called()

    def test_low_confidence_fallback_success(self):
        """Test case where initial low confidence triggers fallback which succeeds."""
        description = "This is a vague description that is also longer than fifty characters so it passes the quality check."
        df = pd.DataFrame({"Description": [description]})

        # Mock predictions:
        # 1. Irrelevant model -> 0
        # 2. Multiclass model -> "Unknown", 0.4 (Low conf)
        # 3. Multiclass model (on summary) -> "Phishing", 0.9 (High conf)
        self.mock_predict.side_effect = [
            [(0, 0.99)],
            [("Unknown", 0.4)],
            [("Phishing", 0.9)],
        ]

        self.mock_clarify.return_value = "Clarified description"

        result_df = inference.inference_pipeline(df)

        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["Predicted Category"], "Phishing")
        self.assertEqual(result_df.iloc[0]["Confidence"], 0.9)

        # Verify Qwen WAS loaded and used
        self.mock_load_qwen.assert_called_once()
        self.mock_clarify.assert_called_once_with(
            description, self.qwen_model, self.qwen_tokenizer, self.mock_device
        )

    def test_low_confidence_fallback_failure(self):
        """Test case where fallback also yields low confidence."""
        description = "This is another very vague description that is long enough to pass the initial data quality validation check."
        df = pd.DataFrame({"Description": [description]})

        # Mock predictions:
        # 1. Irrelevant model -> 0
        # 2. Multiclass model -> "Unknown", 0.3
        # 3. Multiclass model (on summary) -> "Still Unknown", 0.5 (Still low)
        self.mock_predict.side_effect = [
            [(0, 0.99)],
            [("Unknown", 0.3)],
            [("Still Unknown", 0.5)],
        ]

        self.mock_clarify.return_value = "Clarified but still vague"

        result_df = inference.inference_pipeline(df)

        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["Predicted Category"], "Unable to classify")
        self.assertEqual(result_df.iloc[0]["Confidence"], 0.5)

        self.mock_load_qwen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
