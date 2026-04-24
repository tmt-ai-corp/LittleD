#!/usr/bin/env python3
# coding=utf-8
"""Clean regenerated JSONL data before SpecForge chat-template preprocessing.

The expected training path uses the ``conversations`` column and applies a chat
template later. Therefore each conversation message should contain raw text only,
not rendered chat-control tokens such as ``<|turn_end|>``.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CHAT_CONTROL_PATTERNS = [
    re.compile(r"<\|begin_of_text\|>"),
    re.compile(r"<\|end_of_text\|>"),
    re.compile(r"<\|turn_start\|>\s*(system|user|assistant|tool)\s*\n?", re.I),
    re.compile(r"<\|turn_end\|>"),
]

EMPTY_TOOL_PROMPT_RE = re.compile(
    r"\n*You are a function calling AI model\..*?"
    r"Here are the available tools:\s*<tools>\s*</tools>.*?"
    r"</tool_call>\s*",
    re.DOTALL,
)

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
    "user": "user",
    "system": "system",
    "tool": "tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean regenerated JSONL conversations for DFlash training."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--drop-text",
        action="store_true",
        help="Drop the top-level preformatted text field after cleaning conversations.",
    )
    parser.add_argument(
        "--keep-empty-messages",
        action="store_true",
        help="Keep messages whose content becomes empty after cleaning.",
    )
    parser.add_argument(
        "--keep-no-assistant",
        action="store_true",
        help="Keep rows without any non-empty assistant message.",
    )
    parser.add_argument(
        "--clean-text-field",
        action="store_true",
        help="Also clean the top-level text field if it exists.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N cleaned rows to stdout.",
    )
    return parser.parse_args()


def clean_text(text: str, stats: Dict[str, int]) -> Tuple[str, bool]:
    original = text

    text, removed_tool_prompts = EMPTY_TOOL_PROMPT_RE.subn("\n", text)
    stats["empty_tool_prompts_removed"] += removed_tool_prompts

    removed_controls = 0
    for pattern in CHAT_CONTROL_PATTERNS:
        text, count = pattern.subn("", text)
        removed_controls += count
    stats["chat_control_tokens_removed"] += removed_controls

    # Keep paragraph structure, but remove blank space left by deleted prompts.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text, text != original


def normalize_role(message: Dict[str, Any]) -> Optional[str]:
    role = message.get("role", message.get("from"))
    if role is None:
        return None
    return ROLE_MAP.get(str(role).strip().lower(), str(role).strip().lower())


def iter_messages(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        return conversations
    messages = row.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def clean_message(
    message: Dict[str, Any],
    stats: Dict[str, int],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    role = normalize_role(message)
    if role is None:
        stats["messages_dropped_missing_role"] += 1
        return None, True

    raw_content = message.get("content", message.get("value", ""))
    if raw_content is None:
        raw_content = ""

    cleaned = dict(message)
    cleaned.pop("from", None)
    cleaned.pop("value", None)
    cleaned["role"] = role

    changed = cleaned != message
    if isinstance(raw_content, str):
        content, content_changed = clean_text(raw_content, stats)
        cleaned["content"] = content
        changed = changed or content_changed
    else:
        cleaned["content"] = raw_content

    return cleaned, changed


def clean_row(
    row: Dict[str, Any],
    args: argparse.Namespace,
    stats: Dict[str, int],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    cleaned_row = dict(row)
    changed = False
    cleaned_messages: List[Dict[str, Any]] = []

    for message in iter_messages(row):
        if not isinstance(message, dict):
            stats["messages_dropped_malformed"] += 1
            changed = True
            continue

        cleaned_message, message_changed = clean_message(message, stats)
        changed = changed or message_changed
        if cleaned_message is None:
            continue

        content = cleaned_message.get("content", "")
        if (
            not args.keep_empty_messages
            and isinstance(content, str)
            and len(content.strip()) == 0
        ):
            stats["messages_dropped_empty"] += 1
            changed = True
            continue

        cleaned_messages.append(cleaned_message)

    cleaned_row["conversations"] = cleaned_messages
    cleaned_row.pop("messages", None)

    if args.drop_text and "text" in cleaned_row:
        cleaned_row.pop("text", None)
        changed = True
    elif args.clean_text_field and isinstance(cleaned_row.get("text"), str):
        cleaned_text, text_changed = clean_text(cleaned_row["text"], stats)
        cleaned_row["text"] = cleaned_text
        changed = changed or text_changed

    has_assistant = any(
        message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and len(message["content"].strip()) > 0
        for message in cleaned_messages
    )
    if not has_assistant and not args.keep_no_assistant:
        stats["rows_dropped_no_assistant"] += 1
        return None, True

    return cleaned_row, changed


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows_read": 0,
        "rows_written": 0,
        "rows_changed": 0,
        "rows_dropped_malformed_json": 0,
        "rows_dropped_no_assistant": 0,
        "messages_dropped_missing_role": 0,
        "messages_dropped_malformed": 0,
        "messages_dropped_empty": 0,
        "empty_tool_prompts_removed": 0,
        "chat_control_tokens_removed": 0,
    }

    previews: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            stats["rows_read"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["rows_dropped_malformed_json"] += 1
                continue

            cleaned_row, changed = clean_row(row, args, stats)
            if cleaned_row is None:
                continue
            if changed:
                stats["rows_changed"] += 1

            dst.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1
            if len(previews) < args.preview:
                previews.append(cleaned_row)

    for row in previews:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    print("Cleaned JSONL written to:", output_path)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
