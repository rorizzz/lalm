"""Offline prepare cut.conversation for LALM training (Qwen3, no-thinking).

Same as prepare_conversation.py, but renders the assistant turn in Qwen3's
no-thinking format by injecting '<think>\\n\\n</think>\\n\\n' right after
'<|im_start|>assistant\\n'. This matches tokenizer.apply_chat_template(
..., enable_thinking=False) output and is required when the LLM is Qwen3
(Instruct/Thinking) and the LM is frozen during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm
from transformers import AutoTokenizer

from auden.utils.text_normalization import text_normalization

_NO_THINK_BLOCK = "<think>\n\n</think>\n\n"


def _build_conversation(cut, instruction=None, system=None):
    response = cut.supervisions[0].custom.get("answer", cut.supervisions[0].text)
    response = text_normalization(
        response,
        case="lower",
        space_between_cjk=False,
        remove_diacritics=True,
        remove_symbols=True,
        remove_in_parenthesis=True,
        remove_in_brackets=True,
    )

    audio_source = cut.recording.sources[0].source

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    user_content = [{"type": "audio", "audio": audio_source}]
    if instruction:
        user_content.append({"type": "text", "text": str(instruction)})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})
    return messages


def _render_conversation(
    conversation: list[dict], audio_token: str = "<|audio|>"
) -> str:
    """Render conversation to Qwen3-style chat text with no-thinking prefix.

    The assistant turn is rendered as:
        <|im_start|>assistant
        <think>

        </think>

        <answer><|im_end|>
    """
    chunks = []
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if isinstance(content, list):
            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type == "audio":
                    parts.append(audio_token)
                elif item_type == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    raise ValueError(f"Unknown content type: {item_type!r}")
            rendered_content = "".join(parts)
        else:
            rendered_content = str(content)
        if role == "assistant":
            rendered_content = _NO_THINK_BLOCK + rendered_content
        chunks.append(f"<|im_start|>{role}\n{rendered_content}<|im_end|>\n")
    return "".join(chunks)


def _estimate_text_tokens(text: str, tokenizer) -> int:
    text = text.strip()
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def prepare_cut(cut, tokenizer, instruction=None, system=None):
    if cut.custom is None:
        cut.custom = {}
    conversation = _build_conversation(cut, instruction, system)
    rendered_conversation = _render_conversation(conversation)
    cut.custom["conversation"] = conversation
    cut.custom["rendered_conversation"] = rendered_conversation
    cut.custom["num_text_tokens"] = _estimate_text_tokens(
        rendered_conversation,
        tokenizer,
    )
    return cut


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cut.conversation offline for a CutSet (Qwen3 no-thinking)."
    )
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--instruction", help="Additional user prompt")
    parser.add_argument("--system", help="System for the conversation.")
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    out_path = Path(args.output_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    cuts = CutSet.from_file(args.input_manifest)
    prepared = []
    for cut in tqdm(cuts, desc="Preparing conversations (Qwen3 no-thinking)"):
        prepared.append(
            prepare_cut(
                cut,
                tokenizer=tokenizer,
                instruction=args.instruction,
                system=args.system,
            )
        )
    prepared = CutSet.from_cuts(prepared)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_file(str(out_path))
    print(f"Saved prepared CutSet to: {out_path}")


if __name__ == "__main__":
    main()
