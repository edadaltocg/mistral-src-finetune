# Requirements: git-lfs
import concurrent.futures
import os
import subprocess
from pathlib import Path

# HF
root = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
base_url = "https://huggingface.co/"
repos = ["mistralai/Mixtral-8x7B-Instruct-v0.1/", "mistralai/Mixtral-8x7B-v0.1/"]

# Mistral
links_md5 = [
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


def git_lfs(root=root, repos=repos, base_url=base_url):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(repos)) as executor:
        list(
            executor.map(
                subprocess.run,
                (
                    ["git-lfs", "clone", base_url + repo, Path(root) / repo]
                    for repo in repos
                ),
            )
        )


def download_mistral_links_md5(root=root, links_md5=links_md5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(links_md5)) as executor:
        list(
            executor.map(
                subprocess.run,
                (
                    [
                        "wget",
                        "-c",
                        "-O",
                        Path(root) / link_md5[0].split("/")[-1],
                        link_md5[0],
                    ]
                    for link_md5 in links_md5
                ),
            )
        )
        list(
            executor.map(
                subprocess.run,
                (
                    ["md5sum", "-c", "--status", "--strict", Path(root) / link_md5[0]]
                    for link_md5 in links_md5
                ),
            )
        )
        list(
            executor.map(
                subprocess.run,
                (
                    ["tar", "-xvf", Path(root) / link_md5[0].split("/")[-1]]
                    for link_md5 in links_md5
                ),
            )
        )
        list(
            executor.map(
                subprocess.run,
                (
                    ["rm", Path(root) / link_md5[0].split("/")[-1]]
                    for link_md5 in links_md5
                ),
            )
        )


if __name__ == "__main__":
    from fire import Fire

    Fire(download_mistral_links_md5)

    # nohup python scripts/download.py &
