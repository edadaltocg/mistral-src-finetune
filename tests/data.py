import os
import time
from pathlib import Path

import numpy as np
import pytest

from finetune import data


@pytest.mark.parametrize("repo_id", ["tatsu-lab/alpaca"])
def test_download_dataset(repo_id):
    dataset = data.download_hf_dataset(repo_id)
    assert len(dataset) > 0


def test_get_alpaca():
    dataset = data.get_hf_dataset("tatsu-lab/alpaca")
    assert list(dataset["train"].features.keys()) == [
        "instruction",
        "input",
        "output",
        "text",
    ]
    assert len(dataset["train"]) == 52002
    messages = data.alpaca2messages()
    assert len(messages) == 52002 * 2
    examples = data.messages2examples(messages)
    assert set(examples[0].keys()) == set(["prompt", "response", "example"])
    assert len(examples) == 52002


def test_prepare_alpaca():
    data_dir = Path(os.environ["DATA_DIR"])
    model_name = "Mistral-7b-v0.2-instruct"
    repo_id = "tatsu-lab/alpaca"
    data.prepare_alpaca(model_name)
    train_data = np.memmap(data_dir / repo_id / model_name / "train_x.bin", mode="r", dtype=np.int32)
    print(train_data)


def test_build_prompt():
    messages = [
        {"role": "user", "content": "2+2"},
        {"role": "assistant", "content": "4!"},
        {"role": "user", "content": "+2"},
    ]

    prompt_str = data.messages2promptstr(messages)
    print(prompt_str)


def test_dataloader():
    root = Path(os.environ["DATA_DIR"])
    model_name = "Mistral-7b-v0.2-instruct"
    repo_id = "tatsu-lab/alpaca"
    data_dir = root / repo_id / model_name
    batch_size = 2
    block_size = 1024
    dataloader = data.dataloader(
        data_dir, batch_size=batch_size, block_size=block_size, device="cpu", device_type="cpu"
    )
    max_iters = 10
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(dataloader("train")):
        print(x.shape)
        print(y.shape)
        assert x.shape == (batch_size, block_size)
        if i > max_iters:
            break
    t1 = time.perf_counter()
    print(f"Elapsed time: {(t1 - t0):.4f}s ({max_iters/(t1-t0)} it/s)")
    # pytest -vvv tests/data.py::test_dataloader -s
