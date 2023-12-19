import os
from transformers import AutoTokenizer
from typing import List, Dict
from pathlib import Path

TOKENIZERS_DIR = Path(os.environ.get("TOKENIZERS_DIR", "tokenizers"))


def build_prompt(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
):
    prompt = ""
    for i, msg in enumerate(messages):
        is_user = {"user": True, "assistant": False}[msg["role"]]
        assert (i % 2 == 0) == is_user
        content = msg["content"]
        assert content == content.strip()
        if is_user:
            prompt += f"[INST] {content} [/INST]"
        else:
            prompt += f" {content}</s>"
    tokens_ids = tokenizer.encode(prompt)
    token_str = tokenizer.convert_ids_to_tokens(tokens_ids)
    return tokens_ids, token_str


def test_build_prompt():
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tok.save_pretrained(TOKENIZERS_DIR / "mistralai/Mistral-7B-Instruct-v0.2")
    messages = [
        {"role": "user", "content": "2+2"},
        {"role": "assistant", "content": "4!"},
        {"role": "user", "content": "+2"},
        {"role": "assistant", "content": "6!"},
        {"role": "user", "content": "+4"},
    ]

    tokens_ids, token_str = build_prompt(messages, tok)
    print(tokens_ids)
    # [1, 733, 16289, 28793, 28705, 28750, 28806, 28750, 733, 28748, 16289, 28793, 28705, 28781, 28808, 2, 733, 16289, 28793, 648, 28750, 733, 28748, 16289, 28793, 28705, 28784, 28808, 2, 733, 16289, 28793, 648, 28781, 733, 28748, 16289, 28793]
    print(token_str)
    # ['<s>', '▁[', 'INST', ']', '▁', '2', '+', '2', '▁[', '/', 'INST', ']', '▁', '4', '!', '</s>', '▁[', 'INST', ']', '▁+', '2', '▁[', '/', 'INST', ']', '▁', '6', '!', '</s>', '▁[', 'INST', ']', '▁+', '4', '▁[', '/', 'INST', ']']


if __name__ == "__main__":
    test_build_prompt()
