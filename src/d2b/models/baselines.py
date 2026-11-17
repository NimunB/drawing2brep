"""Zero-shot baseline: show test drawings to an untrained Claude and save its replies."""

# Run with: uv run python -m d2b.models.baselines --limit 5 (each drawing costs API credits).

from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path

import anthropic

from d2b.models.prompt import SYSTEM_PROMPT, USER_PROMPT

DEFAULT_PAIRS = Path("data/processed/pairs_test.jsonl")
DEFAULT_MODEL = "claude-opus-5-5"

# Price in US dollars per million tokens: (input, output).
PRICES = {"claude-opus-5-5": (4.00, 20.00)}

# The most tokens a reply may use, counting the model's reasoning.
MAX_TOKENS = 16000

# Fixes the order in which test parts are picked, so a larger run extends a smaller one.
SUBSET_SEED = 0


# Picks which test parts to use: the first `limit` of a fixed shuffle of the test set.
def choose_pairs(pairs_path: Path, limit: int | None) -> list[dict]:
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    random.Random(SUBSET_SEED).shuffle(pairs)
    return pairs[:limit] if limit else pairs


# Works out what one reply cost, in US dollars, from its token counts.
def cost_in_dollars(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


# Sends one drawing to Claude and returns its reply with the token counts.
def ask_claude(client: anthropic.Anthropic, image_path: Path, model: str) -> dict:
    # The image travels inside the request as text (base64).
    image = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    source = {"type": "base64", "media_type": "image/png", "data": image}
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": source},
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    )
    # A reply can hold several blocks (reasoning, then text); only the text is the answer.
    reply = "".join(block.text for block in response.content if block.type == "text")
    return {
        "reply": reply,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# Command-line entry point: ask the model about each chosen drawing and save every reply.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or Path(f"outputs/baseline_{args.model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Replies already saved are not asked for again, so a stopped run can be continued.
    done = set()
    if out.exists():
        done = {json.loads(line)["name"] for line in out.read_text().splitlines()}

    # The client reads the API key from the ANTHROPIC_API_KEY environment variable.
    client = anthropic.Anthropic()
    spent, asked = 0.0, 0
    with out.open("a") as handle:
        for pair in choose_pairs(args.pairs, args.limit):
            if pair["name"] in done:
                continue
            try:
                answer = ask_claude(client, args.pairs.parent / pair["image"], args.model)
            except anthropic.RateLimitError:
                print("rate limited; stopping here. Run again later to continue.")
                break
            except anthropic.APIStatusError as error:
                print(f"API error {error.status_code}: {error.message}; stopping here.")
                break
            except anthropic.APIConnectionError:
                print("network error; stopping here. Run again to continue.")
                break
            cost = cost_in_dollars(args.model, answer["input_tokens"], answer["output_tokens"])
            record = {"name": pair["name"], "model": args.model, "cost": cost} | answer
            # Each reply is saved straight away, so nothing paid for is lost.
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            spent, asked = spent + cost, asked + 1
            tokens = f"{answer['input_tokens']} in, {answer['output_tokens']} out"
            print(f"{pair['name']}: {tokens}, ${cost:.4f}")

    if asked:
        print(f"\nasked: {asked}   spent: ${spent:.2f}   average: ${spent / asked:.4f} per drawing")
    print(f"replies saved to {out}")


if __name__ == "__main__":
    main()
