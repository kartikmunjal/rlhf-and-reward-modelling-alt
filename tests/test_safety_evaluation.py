from pathlib import Path
from unittest.mock import Mock, patch

from scripts.evaluate_safety_classifier import load_safety_model
from src.safety.taxonomy import TARGET_LABELS


@patch("scripts.evaluate_safety_classifier.PeftModel.from_pretrained")
@patch("scripts.evaluate_safety_classifier.AutoModelForSequenceClassification.from_pretrained")
@patch("scripts.evaluate_safety_classifier.PeftConfig.from_pretrained")
def test_model_loader_constructs_locked_three_label_head(
    load_peft_config, load_base, load_adapter
):
    load_peft_config.return_value = Mock(base_model_name_or_path="distilbert-base-uncased")
    base = Mock()
    adapter = Mock()
    load_base.return_value = base
    load_adapter.return_value = adapter

    result = load_safety_model(Path("checkpoint"))

    kwargs = load_base.call_args.kwargs
    assert kwargs["num_labels"] == 3
    assert kwargs["problem_type"] == "multi_label_classification"
    assert kwargs["label2id"] == {name: i for i, name in enumerate(TARGET_LABELS)}
    load_adapter.assert_called_once_with(base, Path("checkpoint"))
    assert result is adapter
