import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, random_split

import mistral
import mistral.tokenizer


# defaults
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TOKEN = os.environ.get("HF_TOKEN")
NUM_PROC = os.cpu_count()
TEMPLATE = "<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]"


def dataloader(data_dir: Path, batch_size: int, block_size: int, device: int | str, device_type: str):
    dtype = np.int32
    train_data_x = np.memmap(os.path.join(data_dir, "train_x.bin"), dtype=dtype, mode="r")
    train_data_y = np.memmap(os.path.join(data_dir, "train_y.bin"), dtype=dtype, mode="r")
    val_data_x = np.memmap(os.path.join(data_dir, "val_x.bin"), dtype=dtype, mode="r")
    val_data_y = np.memmap(os.path.join(data_dir, "val_y.bin"), dtype=dtype, mode="r")

    def get_batch(split):
        data = train_data_x if split == "train" else val_data_x
        labels = train_data_y if split == "train" else val_data_y
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((labels[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        yield x, y

    return get_batch


def download_hf_dataset(repo_id: str, data_dir: Path = DATA_DIR, token: str = TOKEN, num_proc: int = NUM_PROC):
    dataset = load_dataset(repo_id, token=token, num_proc=num_proc)
    dataset.save_to_disk(data_dir / repo_id)
    return dataset


def get_hf_dataset(repo_id: str, data_dir: Path = DATA_DIR, seed: int = 42):
    dataset_path = data_dir / repo_id
    dataset = load_from_disk(str(dataset_path))
    # shuffle
    dataset = dataset.shuffle(seed=seed)
    return dataset


def alpaca2messages(data_dir: Path = DATA_DIR) -> List[Dict[str, str]]:
    """Build messages dataset in memory from Alpaca instructions dataset"""
    messages = []
    dataset = get_hf_dataset("tatsu-lab/alpaca", data_dir=data_dir)

    def preprocess_row(row):
        context = f"{' ' + row['input'] if len(row['input'])>0 else ''}"
        user = f"{row['instruction']}{context}"
        assistant = row["output"]
        return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]

    for row in dataset["train"]:
        messages.extend(preprocess_row(row))

    return messages


def build_prompt(msg: Dict[str, str]) -> str:
    is_user = msg["role"] == "user"
    content = msg["content"]
    assert content == content.strip()
    if is_user:
        return f"[INST] {content} [/INST]"
    else:
        return f" {content}</s>"


def messages2examples(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Break messages into ["prompt", "response", "example"]"""
    examples = []
    prompt = ""
    response = ""
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            prompt = build_prompt(msg)
        if msg["role"] == "assistant":
            response = build_prompt(msg)
        if i % 2 == 1:
            examples.append({"prompt": prompt, "response": response, "example": prompt + response})
    return examples


def messages2promptstr(messages: List[Dict[str, str]]) -> str:
    prompt = ""
    for i, msg in enumerate(messages):
        prompt += build_prompt(msg)
    return prompt


def examples2tokens(
    messages: List[Dict[str, str]], tokenizer: mistral.tokenizer.Tokenizer
) -> Tuple[List[List[int]], List[int]]:
    input_ids, prompt_lens = [], []
    for msg in messages:
        toks = tokenizer.encode(msg["prompt"])
        assert max(toks) < tokenizer.n_words
        prompt_lens.append(len(toks))
        toks = tokenizer.encode(msg["example"])
        input_ids.append(toks)
        assert max(toks) < tokenizer.n_words

    return input_ids, prompt_lens


def get_supervised_labels(toks: List[List[int]], prompt_lens: List[int], ignore_index: int = -1):
    labels = []
    for tok_ids, prompt_len in zip(toks, prompt_lens):
        tok_ids[:prompt_len] = [ignore_index] * prompt_len
        labels.append(tok_ids)
    return labels


def prepare_alpaca(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    checkpoints_dir: str = CHECKPOINTS_DIR,
    data_dir: Path = DATA_DIR,
    val_split_fraction: float = 0.03846,
    seed: int = 42,
):
    """
    Prepare Alpaca instruction dataset.

    repo_id = "tatsu-lab/alpaca"
    Dataset({
        features: ['instruction', 'input', 'output', 'text', 'text_mistral'],
        num_rows: 52002
    })

    The output is a training and val dataset saved as `train.bin` and `val.bin`,
    which stores the preprocessed and tokenized prompts and labels.

    Assume the dataset and the tokenizer are saved to disk.
    """
    # tokenizer
    tokenizer = mistral.tokenizer.Tokenizer(str(checkpoints_dir / model_name / "tokenizer.model"))

    # load dataset
    repo_id = "tatsu-lab/alpaca"
    dataset = get_hf_dataset(repo_id, data_dir=data_dir, seed=seed)
    dataset = alpaca2messages(data_dir=data_dir)
    dataset = messages2examples(dataset)
    input_ids, prompt_lens = examples2tokens(dataset, tokenizer)
    labels = get_supervised_labels(input_ids, prompt_lens)

    train_index = int(len(input_ids) * (1 - val_split_fraction))
    train_input_ids, train_labels = input_ids[:train_index], labels[:train_index]
    val_input_ids, val_labels = input_ids[train_index:], labels[train_index:]

    # flatten
    all_train_input_ids, all_train_labels = [], []
    for x, y in zip(train_input_ids, train_labels):
        all_train_input_ids.extend(x)
        all_train_labels.extend(y)
    assert len(all_train_input_ids) == len(all_train_labels)
    all_val_input_ids, all_val_labels = [], []
    for x, y in zip(val_input_ids, val_labels):
        all_val_input_ids.extend(x)
        all_val_labels.extend(y)
    assert len(all_val_input_ids) == len(all_val_labels)

    # save to disk
    dest_folder = data_dir / repo_id / model_name
    dtype = np.int32
    os.makedirs(dest_folder, exist_ok=True)
    np.array(all_train_input_ids, dtype=dtype).tofile(dest_folder / "train_x.bin")
    np.array(all_train_labels, dtype=dtype).tofile(dest_folder / "train_y.bin")
    np.array(all_val_input_ids, dtype=dtype).tofile(dest_folder / "val_x.bin")
    np.array(all_val_labels, dtype=dtype).tofile(dest_folder / "val_y.bin")


def download_mmlu():
    # TODO
    repo_id = "lukaemon/mmlu"
    names = [
        "high_school_european_history",
        "business_ethics",
        "clinical_knowledge",
        "medical_genetics",
        "high_school_us_history",
        "high_school_physics",
        "high_school_world_history",
        "virology",
        "high_school_microeconomics",
        "econometrics",
        "college_computer_science",
        "high_school_biology",
        "abstract_algebra",
        "professional_accounting",
        "philosophy",
        "professional_medicine",
        "nutrition",
        "global_facts",
        "machine_learning",
        "security_studies",
        "public_relations",
        "professional_psychology",
        "prehistory",
        "anatomy",
        "human_sexuality",
        "college_medicine",
        "high_school_government_and_politics",
        "college_chemistry",
        "logical_fallacies",
        "high_school_geography",
        "elementary_mathematics",
        "human_aging",
        "college_mathematics",
        "high_school_psychology",
        "formal_logic",
        "high_school_statistics",
        "international_law",
        "high_school_mathematics",
        "high_school_computer_science",
        "conceptual_physics",
        "miscellaneous",
        "high_school_chemistry",
        "marketing",
        "professional_law",
        "management",
        "college_physics",
        "jurisprudence",
        "world_religions",
        "sociology",
        "us_foreign_policy",
        "high_school_macroeconomics",
        "computer_security",
        "moral_scenarios",
        "moral_disputes",
        "electrical_engineering",
        "astronomy",
        "college_biology",
    ]

    def download(name):
        return load_dataset(repo_id, token=TOKEN, num_proc=1, name=name)

    with ThreadPoolExecutor(max_workers=len(names) // 2) as exec:
        datasets = list(exec.map(download, names))
    # concat datasets and save to disk
    dataset = datasets[0].concatenate(datasets[1:])  # BUG: : 'DatasetDict' object has no attribute 'concatenate'

    dataset.save_to_disk(DATA_DIR / repo_id)
