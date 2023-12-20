from finetune import data
import mistral
import pytest
from pathlib import Path


def test_download():
    data._download_alpaca()


def test_get_dataset():
    dataset = data.get_alpaca()
    assert list(dataset["train"].features.keys()) == [
        "instruction",
        "input",
        "output",
        "text",
    ]
    assert len(dataset["train"]) == 52002


def test_prepare_alpaca():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset = data.prepare_alpaca(model_name)
    print(dataset)
