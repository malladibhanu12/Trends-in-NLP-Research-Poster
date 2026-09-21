"""
Prompt templates for binary sentiment classification.

Conditions:
- Zero-shot: requests a single sentiment label.
- Few-shot: provides six demonstrations, balanced by sentiment.
- Chain-of-thought: requests reasoning and an explicit final label.

The parser uses the last explicit final-label line when present.
Otherwise, it accepts only a standalone sentiment label.
Invalid or neutral responses are returned as 'Unclear'.
"""

import re


def zero_shot_prompt(text: str) -> str:
    """Build the zero-shot classification prompt."""
    return f"""Classify the sentiment of the following text as exactly one word: Positive or Negative.
Respond with ONLY the single word, nothing else.

Text: "{text}"

Sentiment:"""


def few_shot_prompt(text: str) -> str:
    """Build a prompt with three positive and three negative examples."""
    return f"""Classify the sentiment of the following text as exactly one word: Positive or Negative.
Respond with ONLY the single word, nothing else.

Examples:
Text: "The pianist gave a beautiful performance that moved me to tears."
Sentiment: Positive

Text: "The shoes pinched my toes and left painful blisters."
Sentiment: Negative

Text: "Despite my initial doubts, the workshop was engaging and useful."
Sentiment: Positive

Text: "How thoughtful of them to print my name incorrectly on the award."
Sentiment: Negative

Text: "That guitarist absolutely killed it; the crowd cheered for an encore."
Sentiment: Positive

Text: "The replacement battery barely lasts an hour, so I regret buying it."
Sentiment: Negative

Now classify this text:
Text: "{text}"
Sentiment:"""


def cot_prompt(text: str) -> str:
    """Build the reasoning prompt with an explicit final-answer format."""
    return f"""Classify the sentiment of the following text as Positive or Negative.
Think through it step by step, then give your final answer on the LAST line
in the exact format "Final: Positive" or "Final: Negative".

Steps to follow:
1. Identify emotional or evaluative words/phrases in the text.
2. Check for negation (words like "not", "n't", "never") and how they affect meaning.
3. Check for sarcasm or irony (e.g. positive-sounding words used to mean the opposite).
4. Combine these observations into an overall sentiment judgment.

Text: "{text}"

Reasoning:"""


def parse_final_label(raw_response: str) -> str:
    """
    Extract a sentiment label using a fixed parsing rule.

    Rules:
    1. Restore newlines represented by ' | ' in saved result files.
    2. If explicit 'Final:' lines exist, inspect the last one.
       Accept it only if its value is Positive or Negative.
    3. Otherwise, accept only a standalone Positive or Negative response.
    4. Return 'Unclear' for neutral, ambiguous, or malformed answers.

    Matching is case-insensitive. Sentiment words inside explanations
    are not treated as predictions.
    """
    if not isinstance(raw_response, str):
        return "Unclear"

    text = raw_response.replace(" | ", "\n").strip()

    if not text:
        return "Unclear"

    # Use the last explicit final-answer line, not the first.
    final_lines = re.findall(
        r"^[ \t]*Final[ \t]*:[ \t]*([^\r\n]*)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if final_lines:
        candidate = final_lines[-1].strip()
    else:
        candidate = text

    # Reject explanations, multiple labels, and out-of-task labels.
    if re.fullmatch(r"Positive|Negative", candidate, flags=re.IGNORECASE):
        return candidate.capitalize()

    return "Unclear"