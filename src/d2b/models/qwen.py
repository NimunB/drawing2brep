"""Zero-shot baseline with Qwen2.5-VL: show it test drawings and save the programs it writes."""

# Run on the GPU machine with: uv run --extra gpu python -m d2b.models.qwen --limit 243
# Add --adapter outputs/finetune/RUN/checkpoint-N to use weights trained by finetune.py.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from d2b.models.baselines import DEFAULT_PAIRS, choose_pairs
from d2b.models.prompt import SYSTEM_PROMPT, USER_PROMPT

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# The most tokens the model may write for one drawing.
MAX_NEW_TOKENS = 1024

# How many drawings the model works on at once.
BATCH_SIZE = 8

# Our drawings are 768 x 768 pixels; this stops the model shrinking them before it looks.
MAX_PIXELS = 768 * 768
MIN_PIXELS = 256 * 28 * 28


# Builds the conversation sent for one drawing: the instructions, then the image and request.
def build_messages() -> list[dict]:
    # The image itself is passed separately; this entry only marks where it goes.
    user_content = [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# Lists the pairs that still need a reply, so a stopped run continues where it left off.
def pending_pairs(pairs: list[dict], out: Path) -> list[dict]:
    if not out.exists():
        return pairs
    done = {json.loads(line)["name"] for line in out.read_text().splitlines()}
    return [pair for pair in pairs if pair["name"] not in done]


# Splits a list into groups of at most `size`.
def batches(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# Picks where the model runs: an Nvidia GPU if there is one, else an Apple GPU, else the CPU.
def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Names the replies file after the model, and after the fine-tuning run if one is used.
def default_out(model_id: str, adapter: Path | None) -> Path:
    if adapter is None:
        return Path(f"outputs/baseline_{model_id.split('/')[-1]}.jsonl")
    # An adapter sits at .../RUN/checkpoint-N, so both parts go into the name.
    return Path(f"outputs/finetuned_{adapter.parent.name}_{adapter.name}.jsonl")


# Adds fine-tuned LoRA weights, saved by finetune.py, on top of a loaded model.
def add_adapter(model, adapter: Path):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter)).eval()


# Downloads (the first time) and loads the model and the processor that prepares its inputs.
def load_model(model_id: str, device: str):
    # These libraries are only installed on the GPU machine, so they are imported here.
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Half-size numbers halve the memory needed and are the form the weights are stored in.
    dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    # When replies are generated in a batch, shorter prompts must be padded on the left.
    processor.tokenizer.padding_side = "left"
    return model, processor


# Asks the model about a batch of drawings; returns one record per drawing.
def generate_replies(model, processor, images: list[Image.Image], max_new_tokens: int) -> list:
    import torch

    # Every drawing gets the same conversation, turned into the text format the model expects.
    text = processor.apply_chat_template(
        build_messages(), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text] * len(images), images=images, padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        # do_sample=False always picks the most likely next token, so runs can be repeated.
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # The output starts with a copy of the prompt; only what comes after it is the reply.
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    replies = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # A reply that used every allowed token was probably cut off before it finished.
    ended = processor.tokenizer.eos_token_id
    padding = processor.tokenizer.pad_token_id
    records = []
    for reply, tokens in zip(replies, new_tokens, strict=True):
        finished = any(int(t) in (ended, padding) for t in tokens)
        records.append({"reply": reply, "cut_off": not finished})
    return records


# Command-line entry point: ask Qwen about each chosen drawing and save every reply.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or default_out(args.model, args.adapter)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The same fixed shuffle as the Claude baseline, so both models see the same parts.
    todo = pending_pairs(choose_pairs(args.pairs, args.limit), out)
    print(f"drawings to do: {len(todo)}")
    if not todo:
        return

    device = pick_device()
    print(f"loading {args.model} on {device} ...")
    model, processor = load_model(args.model, device)
    if args.adapter:
        model = add_adapter(model, args.adapter)

    started, done = time.time(), 0
    with out.open("a") as handle:
        for batch in batches(todo, args.batch_size):
            images = [Image.open(args.pairs.parent / p["image"]).convert("RGB") for p in batch]
            answers = generate_replies(model, processor, images, args.max_new_tokens)
            for pair, answer in zip(batch, answers, strict=True):
                record = {"name": pair["name"], "model": args.model} | answer
                if args.adapter:
                    record["adapter"] = str(args.adapter)
                handle.write(json.dumps(record) + "\n")
            # Saved after every batch, so a stopped run loses at most one batch.
            handle.flush()
            done += len(batch)
            rate = done / (time.time() - started)
            print(f"{done} of {len(todo)} done ({rate:.2f} drawings per second)")

    print(f"replies saved to {out}")


if __name__ == "__main__":
    main()
