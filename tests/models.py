from pathlib import Path

import pytest
import torch

import finetune
import mistral
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer


@pytest.mark.parametrize("model_name", set("Mistral-7b-v0.2-instruct"))
@torch.inference_mode()
def test_inference(model_name):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    model = Transformer.from_folder(Path(model_path), max_batch_size=3, device="cuda")
    model.eval()

    prompts = [
        "This is a test",
        "This is another great test",
        "This is a third test, mistral AI is very good at testing. ",
    ]
    inst_prompts = [f"[INST] {p} [/INST]" for p in prompts]
    encoded_prompts = [tokenizer.encode(p, bos=True) for p in inst_prompts]
    print(encoded_prompts)
