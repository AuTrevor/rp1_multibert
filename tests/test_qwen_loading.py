import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference


class TestQwenLoading(unittest.TestCase):
    def setUp(self):
        self.patcher_exists = patch("os.path.exists")
        self.mock_exists = self.patcher_exists.start()

        self.patcher_tokenizer = patch("inference.AutoTokenizer")
        self.mock_tokenizer_cls = self.patcher_tokenizer.start()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer_cls.from_pretrained.return_value = self.mock_tokenizer

        self.patcher_model = patch("inference.AutoModelForImageTextToText")
        self.mock_model_cls = self.patcher_model.start()
        self.mock_model = MagicMock()
        self.mock_model.device = "cpu"
        self.mock_model_cls.from_pretrained.return_value = self.mock_model

        self.qwen_id = inference.QWEN_MODEL_ID
        self.qwen_path = inference.QWEN_MODEL_PATH

    def tearDown(self):
        self.patcher_exists.stop()
        self.patcher_tokenizer.stop()
        self.patcher_model.stop()

    def test_load_local_exists(self):
        """Test loading when local model exists."""
        self.mock_exists.return_value = True

        inference.load_qwen_model()

        # Should load from local path
        self.mock_tokenizer_cls.from_pretrained.assert_called_with(
            self.qwen_path, trust_remote_code=True
        )
        self.mock_model_cls.from_pretrained.assert_called_with(
            self.qwen_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        # Should NOT save
        self.mock_tokenizer.save_pretrained.assert_not_called()
        self.mock_model.save_pretrained.assert_not_called()

    def test_load_local_missing(self):
        """Test loading when local model is missing."""
        self.mock_exists.return_value = False

        inference.load_qwen_model()

        # Should load from Hub ID
        self.mock_tokenizer_cls.from_pretrained.assert_called_with(
            self.qwen_id, trust_remote_code=True
        )
        self.mock_model_cls.from_pretrained.assert_called_with(
            self.qwen_id, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        )

        # Should SAVE to local path
        self.mock_tokenizer.save_pretrained.assert_called_with(self.qwen_path)
        self.mock_model.save_pretrained.assert_called_with(self.qwen_path)


if __name__ == "__main__":
    unittest.main()
