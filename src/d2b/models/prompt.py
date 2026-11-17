"""The instructions given to a model, and how its reply is turned back into a program."""

# The same prompt is used for every model, so that their scores can be compared.

from __future__ import annotations

import re

# Instructions that describe the drawing and the kind of program wanted. No examples are given.
SYSTEM_PROMPT = """\
You convert technical drawings of mechanical parts into CadQuery programs.

The drawing shows one part in three views, all at the same scale:
- FRONT (lower left): the part seen from the front. Across is x, up is z.
- TOP (above the front view): the part seen from above. Across is x, up is y, so the back of \
the part is at the top of this view.
- RIGHT (to the right of the front view): the part seen from its right side. Across is y, so \
the front of the part is at the left of this view. Up is z.

Solid lines are visible edges. Dashed lines are hidden edges. The blue numbers are the part's \
overall width (x), height (z) and depth (y), in centimetres. No other sizes are written. \
Estimate them from the drawing, in proportion to the overall sizes.

Write a Python program that builds this part with CadQuery.
- Import only cadquery (as cq) and, if needed, math.
- Store the finished solid in a variable named result.
- Work in centimetres.
- Keep the orientation shown: x is width, y is depth, z is height. Where the part sits in \
space does not matter.
- Do not read or write files, and do not display anything.

Reply with the program in a single ```python code block and nothing else."""

# The short request sent with each drawing.
USER_PROMPT = "Write the CadQuery program for the part in this drawing."

# Matches a fenced code block, with or without the word "python" after the opening fence.
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


# Pulls the program out of a model's reply; returns None if there is no code block.
def extract_program(reply: str) -> str | None:
    blocks = CODE_BLOCK.findall(reply)
    # If the model wrote several blocks, the last one is its final answer.
    return blocks[-1].strip() + "\n" if blocks else None
