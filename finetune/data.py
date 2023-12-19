import sys
import datasets
import torch
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import mistral


TEMPLATE = (
    "<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]"
)
TOKEN = os.environ.get("HF_TOKEN")
NUM_PROC = os.cpu_count()
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def poorsmans_dataloader(dataset, batch_size, block_size, device, device_type):
    root = os.environ.get("DATA_DIR", "data")
    data_dir = os.path.join(root, dataset)
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def _download_alpaca():
    repo_id = "tatsu-lab/alpaca"
    dataset = load_dataset(repo_id, token=TOKEN, num_proc=NUM_PROC)
    dataset.save_to_disk(DATA_DIR / repo_id)


def _download_mmlu():
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
    dataset = datasets[0].concatenate(
        datasets[1:]
    )  # BUG: : 'DatasetDict' object has no attribute 'concatenate'

    dataset.save_to_disk(DATA_DIR / repo_id)


def download_datasets():
    _download_alpaca()
    _download_mmlu()


def prepare_dataset(
    repo_id: str,
    tokenizer: mistral.tokenizer.Tokenizer,
    prompt_template: str,
):
    dataset = load_dataset(repo_id, token=os.environ.get("HF_TOKEN"))
    data = dataset["train"]


def prepare_alpaca(
    # tokenizer: mistral.tokenizer.Tokenizer,
):
    """
    Prepare Alpaca instruction dataset.

    repo_id = "tatsu-lab/alpaca"


    Dataset({
        features: ['instruction', 'input', 'output', 'text', 'text_mistral'],
        num_rows: 52002
    })

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    os.makedirs(data_dir, exist_ok=True)
    print("Data dir", data_dir)
    repo_id = "tatsu-lab/alpaca"
    dataset = datasets.load_from_disk(data_dir / repo_id)

    # bos_tok = tokenizer.bos_id
    # eos_tok = tokenizer.eos_id
    boi_tok = "[INST]"
    eoi_tok = "[/INST]"
    # add new column to the dataset with the formatted instructions
    dataset = dataset.map(
        lambda x: {
            "inst_mistral_x": f"{boi_tok}{x['instruction']}{' ' + x['input'] if len(x['input'])>0 else ''}{eoi_tok}",
            "inst_mistral_y": f"{x['output']}",
        }
    )

    dataset.save_to_disk(data_dir / (repo_id + "_mistral"))
    print("Example:", dataset["train"][0])

    def prepare_sample(
        example: dict,
        tokenizer: Tokenizer,
        max_length: int,
        mask_inputs: bool,
        ignore_index: int,
    ) -> dict:
        """Processes a single sample.

        Each sample in the dataset consists of:
        - instruction: A string describing the task
        - input: A string holding a special input value for the instruction.
            This only applies to some samples, and in others this is empty.
        - output: The response string

        This function processes this data to produce a prompt text and a label for
        supervised training. The prompt text is formed as a single message including both
        the instruction and the input. The label/target is the same message but with the
        response attached.

        Finally, both the prompt and the label get tokenized. If desired, all tokens
        in the label that correspond to the original input prompt get masked out (default).
        """
        full_prompt = generate_prompt(example)
        full_prompt_and_response = full_prompt + example["output"]
        encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
        encoded_full_prompt_and_response = tokenizer.encode(
            full_prompt_and_response, eos=True, max_length=max_length
        )

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_full_prompt_and_response.clone()
        if mask_inputs:
            labels[: len(encoded_full_prompt)] = ignore_index

        return {
            **example,
            "input_ids": encoded_full_prompt_and_response,
            "labels": labels,
        }

    # tokenize
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    return dataset


def prepare_mmlu():
    return


def prepare_datasets():
    prepare_alpaca()
    prepare_mmlu()


if __name__ == "__main__":
    download_datasets()
    prepare_datasets()
