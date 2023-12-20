# Requirements: git-lfs
from typing import List, Tuple
import concurrent.futures
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))

# HF
HF_BASE_URL = "https://huggingface.co/"
HF_REPOS = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1/",
    "mistralai/Mixtral-8x7B-v0.1/",
]

# Mistral
MISTRAL_LINKS_MD5 = [
    (
        "https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar",
        "37dab53973db2d56b2da0a033a15307f",
    ),
    (
        "https://files.mistral-7b-v0-2.mistral.ai/Mistral-7B-v0.2-Instruct.tar",
        "fbae55bc038f12f010b4251326e73d39",
    ),
    (
        "https://files.mixtral-8x7b-v0-1.mistral.ai/Mixtral-8x7B-v0.1-Instruct.tar",
        "8e2d3930145dc43d3084396f49d38a3f",
    ),
]


def git_lfs(
    root: Path = ROOT, repos: List[str] = HF_REPOS, base_url: str = HF_BASE_URL
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(repos)) as executor:
        list(
            executor.map(
                subprocess.run,
                (["git-lfs", "clone", base_url + repo, root / repo] for repo in repos),
            )
        )


def download_md5_check_extract_rm(link: str, md5: str, root: Path = ROOT):
    print("Downloading", link)
    filename = link.split("/")[-1].capitalize()
    subprocess.run(["wget", "-c", "-O", root / filename, link])
    result = subprocess.run(
        ["md5sum", "-c", "--status", "--strict", root / filename],
        capture_output=True,
        text=True,
    )
    md5sum = result.stdout.split()[0]
    assert md5 == md5sum, f"{md5} != {md5sum}"
    subprocess.run(["tar", "-xvf", root / filename])
    subprocess.run(["rm", root / filename])
    assert not (root / filename).exists()


def download_links_md5(
    links_md5: List[Tuple[str, str]] = MISTRAL_LINKS_MD5, root: Path = ROOT
):
    args = [(lm[0], lm[1], root) for lm in links_md5]
    print("Args", args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(links_md5)) as executor:
        list(executor.map(lambda x: download_md5_check_extract_rm(*x), args))


if __name__ == "__main__":
    download_links_md5([MISTRAL_LINKS_MD5[1]])
    # download_links_md5()
    # nohup python scripts/download.py &
