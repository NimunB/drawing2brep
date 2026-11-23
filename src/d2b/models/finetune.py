"""Fine-tune Qwen2.5-VL with LoRA to write the CadQuery program for a drawing."""

# Run on the GPU machine with: uv run --extra gpu python -m d2b.models.finetune --run-name first

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from PIL import Image

from d2b.models.qwen import DEFAULT_MODEL, batches, build_messages, pick_device

DEFAULT_TRAIN_PAIRS = Path("data/processed/pairs_train.jsonl")
DEFAULT_OUT = Path("outputs/finetune")

# Training pairs held back to watch for overfitting; they are never trained on.
VALIDATION_COUNT = 300

# Fixes which pairs are held back and the order of training, so a run can be repeated.
SEED = 0

# Labels longer than this many tokens are left out of training (about 3% of them).
MAX_ANSWER_TOKENS = 2048

# The value PyTorch's loss ignores; it marks tokens the model is not trained to write.
IGNORE = -100

# The text that opens the model's turn; everything after it is the answer.
ANSWER_MARKER = "<|im_start|>assistant\n"

# The layers that get LoRA weights: the attention layers of the language part of the model.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# Wraps a program in the code block the prompt asks for, so training matches the baselines.
def format_answer(program: str) -> str:
    return f"```python\n{program.strip()}\n```"


# Builds the full conversation for one training pair: instructions, drawing, then the answer.
def build_training_messages(program: str) -> list[dict]:
    return [*build_messages(), {"role": "assistant", "content": format_answer(program)}]


# Splits the pairs into a training set and a held-back validation set, the same way every time.
def split_validation(pairs: list[dict], count: int, seed: int = SEED) -> tuple[list, list]:
    shuffled = sorted(pairs, key=lambda pair: pair["name"])
    random.Random(seed).shuffle(shuffled)
    return shuffled[count:], shuffled[:count]


# Keeps the pairs whose answer is no longer than the limit; `lengths` gives each one's tokens.
def keep_short(pairs: list[dict], lengths: list[int], limit: int) -> list[dict]:
    return [pair for pair, length in zip(pairs, lengths, strict=True) if length <= limit]


# Finds where the answer begins: just after the last place the marker appears in the tokens.
def find_answer_start(tokens: list[int], marker: list[int]) -> int:
    for start in range(len(tokens) - len(marker), -1, -1):
        if tokens[start : start + len(marker)] == marker:
            return start + len(marker)
    raise ValueError("the answer marker is not in the tokens")


# Makes the training targets for one example: the answer's tokens, with everything else ignored.
def mask_labels(tokens: list[int], marker: list[int], padding: int) -> list[int]:
    start = find_answer_start(tokens, marker)
    # The instructions and the drawing are context; padding is filler added to equalise lengths.
    return [
        IGNORE if position < start or token == padding else token
        for position, token in enumerate(tokens)
    ]


# Scales the learning rate: it climbs from 0 during warm-up, then falls smoothly back to 0.
def learning_rate_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# Counts the tokens in each pair's answer.
def answer_lengths(processor, pairs: list[dict]) -> list[int]:
    answers = [format_answer(pair["program"]) for pair in pairs]
    return [len(ids) for ids in processor.tokenizer(answers, add_special_tokens=False).input_ids]


# Loads the model with its own weights frozen and small trainable LoRA weights attached.
def load_for_training(model_id: str, device: str, rank: int, alpha: int, dropout: float):
    # These libraries are only installed on the GPU machine, so they are imported here.
    import torch
    from peft import LoraConfig, get_peft_model

    from d2b.models.qwen import load_model

    model, processor = load_model(model_id, device)
    # In training, shorter examples are padded at the end, so every answer starts in context.
    processor.tokenizer.padding_side = "right"
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    # The frozen weights stay half-size; the few trained ones are full-size for accurate updates.
    for _, parameter in trainable:
        parameter.data = parameter.data.to(torch.float32)
    in_image_part = sum("visual" in name for name, _ in trainable)
    count = sum(p.numel() for _, p in trainable)
    print(f"trainable weights: {count:,} in {len(trainable)} tensors")
    print(f"of those tensors, in the image part: {in_image_part}")
    return model.train(), processor


# Turns a batch of pairs into the tensors the model trains on: inputs and masked targets.
def encode_batch(processor, pairs: list[dict], image_root: Path, device: str) -> dict:
    import torch

    images = [Image.open(image_root / pair["image"]).convert("RGB") for pair in pairs]
    texts = [
        processor.apply_chat_template(build_training_messages(pair["program"]), tokenize=False)
        for pair in pairs
    ]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    tokenizer = processor.tokenizer
    marker = tokenizer(ANSWER_MARKER, add_special_tokens=False).input_ids
    labels = [mask_labels(row, marker, tokenizer.pad_token_id) for row in inputs.input_ids.tolist()]
    inputs["labels"] = torch.tensor(labels)
    return inputs.to(device)


# Runs the model on one batch and returns its loss and how many answer tokens it covered.
def batch_loss(model, processor, pairs: list[dict], image_root: Path, device: str):
    import torch

    inputs = encode_batch(processor, pairs, image_root, device)
    tokens = int((inputs["labels"] != IGNORE).sum())
    # Half-size arithmetic on the GPU, which is faster and is what the frozen weights use.
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        loss = model(**inputs).loss
    return loss, tokens


# Measures the average loss per answer token over the validation pairs, without training.
def validation_loss(model, processor, pairs: list[dict], image_root: Path, device: str, size: int):
    import torch

    # Dropout is switched off while measuring, and back on afterwards.
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in batches(pairs, size):
            loss, count = batch_loss(model, processor, batch, image_root, device)
            # Each batch counts in proportion to its tokens, so long answers are not under-weighted.
            total += float(loss) * count
            tokens += count
    model.train()
    return total / tokens


# Trains the model, checking the validation loss and saving a checkpoint at regular intervals.
def train(model, processor, train_pairs, validation_pairs, args, run_dir: Path) -> None:
    import torch

    device = pick_device()
    image_root = args.pairs.parent
    weights = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(weights, lr=args.learning_rate, weight_decay=0.0)
    # One update uses several small batches added together, to act like one larger batch.
    updates_per_pass = math.ceil(len(train_pairs) / (args.batch_size * args.accumulate))
    total = updates_per_pass * args.epochs
    if args.max_updates:
        total = min(total, args.max_updates)
    warmup = max(1, int(total * args.warmup))
    print(f"training pairs: {len(train_pairs)}, updates: {total}, warm-up: {warmup}")

    log_path = run_dir / "log.jsonl"
    best = {"validation_loss": float("inf"), "update": 0}
    update, started = 0, time.time()
    running, running_tokens = 0.0, 0
    for epoch in range(args.epochs):
        # A different order on every pass, fixed by the seed.
        order = train_pairs[:]
        random.Random(args.seed + epoch).shuffle(order)
        for group in batches(batches(order, args.batch_size), args.accumulate):
            # The learning rate for this update comes from the warm-up and decay schedule.
            for scale in optimizer.param_groups:
                scale["lr"] = args.learning_rate * learning_rate_scale(update, total, warmup)
            for batch in group:
                loss, tokens = batch_loss(model, processor, batch, image_root, device)
                # Each small batch contributes its share of the gradient.
                (loss / len(group)).backward()
                running += float(loss) * tokens
                running_tokens += tokens
            # Very large gradients are scaled down, so one odd batch cannot wreck the weights.
            torch.nn.utils.clip_grad_norm_(weights, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1

            if update % args.check_every == 0 or update == total:
                record = {
                    "update": update,
                    "epoch": epoch + 1,
                    "train_loss": running / running_tokens,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "minutes": (time.time() - started) / 60.0,
                }
                running, running_tokens = 0.0, 0
                if validation_pairs:
                    record["validation_loss"] = validation_loss(
                        model, processor, validation_pairs, image_root, device, args.batch_size
                    )
                    # Remember which checkpoint had the lowest validation loss so far.
                    if record["validation_loss"] < best["validation_loss"]:
                        best = {"validation_loss": record["validation_loss"], "update": update}
                        (run_dir / "best.json").write_text(json.dumps(best))
                # Only the small LoRA weights are saved, not the whole model.
                model.save_pretrained(run_dir / f"checkpoint-{update}")
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record))
            if update >= total:
                print(f"finished; lowest validation loss: {best}")
                return


# Command-line entry point: read the pairs, split them, load the model and train.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_TRAIN_PAIRS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-count", type=int, default=VALIDATION_COUNT)
    parser.add_argument("--max-answer-tokens", type=int, default=MAX_ANSWER_TOKENS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    run_dir = args.out / args.run_name
    if (run_dir / "log.jsonl").exists():
        raise SystemExit(f"{run_dir} already holds a run; choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    # The settings are saved beside the results, so the run can be repeated.
    settings = {key: str(value) for key, value in vars(args).items()}
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines()]
    # The validation pairs are split off first, so they are the same for every run.
    train_pairs, validation_pairs = split_validation(pairs, args.validation_count, args.seed)
    if validation_pairs:
        # Saved so the Qwen script can later write programs for these same drawings.
        validation_path = args.pairs.parent / "pairs_validation.jsonl"
        validation_path.write_text("".join(json.dumps(p) + "\n" for p in validation_pairs))
    if args.limit:
        train_pairs = train_pairs[: args.limit]

    device = pick_device()
    print(f"loading {args.model} on {device} ...")
    model, processor = load_for_training(args.model, device, args.rank, args.alpha, args.dropout)

    # Very long answers are left out: a few of them would set how much memory training needs.
    kept = keep_short(train_pairs, answer_lengths(processor, train_pairs), args.max_answer_tokens)
    print(f"left out for length: {len(train_pairs) - len(kept)} of {len(train_pairs)}")
    lengths = answer_lengths(processor, validation_pairs) if validation_pairs else []
    validation_pairs = keep_short(validation_pairs, lengths, args.max_answer_tokens)

    train(model, processor, kept, validation_pairs, args, run_dir)
    print(f"checkpoints and log saved to {run_dir}")


if __name__ == "__main__":
    main()
