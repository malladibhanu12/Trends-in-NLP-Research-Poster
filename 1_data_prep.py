"""
STEP 1: Build the labeled dataset for the experiment.

Produces dataset.csv with columns:
    id, text, gold_label (Positive/Negative), category (clear/ambiguous)

- 60 "clear" sentences: sampled from NLTK's Twitter Samples corpus
  (Bird, Klein & Loper, "Natural Language Processing with Python", O'Reilly,
  2009) -- well-known, pre-labeled positive/negative tweets.
- 40 "ambiguous" sentences: hand-curated, covering negation, sarcasm, and
  slang, with manually verified gold labels. These are intentionally
  designed to be hard for both rule-based and LLM methods.

Run this once. It will download the small NLTK twitter_samples corpus
(a few MB) the first time.
"""

import random
import csv
import nltk

random.seed(42)  # reproducibility

# ---- Step 1: download NLTK data (only happens once) ----
try:
    nltk.data.find("corpora/twitter_samples")
except LookupError:
    print("Downloading NLTK twitter_samples corpus (one-time, ~3MB)...")
    nltk.download("twitter_samples")

from nltk.corpus import twitter_samples

# ---- Step 2: sample 30 positive + 30 negative "clear" tweets ----
pos_tweets = twitter_samples.strings("positive_tweets.json")
neg_tweets = twitter_samples.strings("negative_tweets.json")


def clean_tweet(t):
    """Light cleanup: remove very short/garbled tweets, strip whitespace."""
    t = t.strip().replace("\n", " ")
    return t


pos_clean = [clean_tweet(t) for t in pos_tweets if len(t) > 20 and len(t) < 140]
neg_clean = [clean_tweet(t) for t in neg_tweets if len(t) > 20 and len(t) < 140]

random.shuffle(pos_clean)
random.shuffle(neg_clean)

clear_positive = pos_clean[:30]
clear_negative = neg_clean[:30]

clear_rows = (
    [{"text": t, "gold_label": "Positive", "category": "clear"} for t in clear_positive]
    + [{"text": t, "gold_label": "Negative", "category": "clear"} for t in clear_negative]
)

# ---- Step 3: hand-curated ambiguous sentences (20 positive, 20 negative) ----
# These test negation, sarcasm, slang, and mixed sentiment -- cases where
# simple lexicon matching (VADER) and naive LLM prompting are expected to
# struggle. Gold labels reflect the INTENDED meaning, not surface polarity.

ambiguous_positive = [
    "I don't think this could have gone any better.",
    "Not bad at all, actually really impressed.",
    "I can't complain, this exceeded expectations.",
    "The service was far from disappointing.",
    "That's fire, honestly one of the best I've tried.",
    "Never thought I'd say this, but I loved it.",
    "It's not the worst thing I've ever eaten, actually quite good.",
    "I wasn't expecting much, but wow, pleasantly surprised.",
    "Can't say I hate it -- it's actually pretty solid.",
    "This is sick! Best purchase in a while.",
    "Honestly didn't expect to enjoy it this much.",
    "Not gonna lie, this slaps.",
    "I have zero complaints, genuinely great experience.",
    "Who knew something so cheap could be this good.",
    "I take back what I said earlier, this is actually decent.",
    "The reviews undersell it, this is genuinely great.",
    "It's not perfect, but I really can't find much to complain about.",
    "Didn't think I'd say this about a Monday, but today was great.",
    "This app doesn't crash anymore, finally something that works.",
    "I'm not even mad, this turned out better than planned.",
]

ambiguous_negative = [
    "Oh great, another Monday, just what I needed.",
    "Wow, my flight got delayed again, love that for me.",
    "I don't think the movie is terrible, it's actually much worse than that.",
    "Sure, take your time, it's not like I have anywhere to be.",
    "This is exactly the kind of 'quality' I expected.",
    "Fantastic, the app crashed again right before the deadline.",
    "Nothing says 'relaxing weekend' like a broken washing machine.",
    "Yeah, because waiting three hours on hold is such a treat.",
    "The food wasn't good, and the service wasn't much better either.",
    "I'm sure the extra fees were 'totally worth it'.",
    "Just what I always wanted, more spam emails.",
    "Great, now the wifi is down too. Perfect timing.",
    "I guess losing my luggage really 'made my trip'.",
    "Oh sure, because customer support has been SO responsive.",
    "This is not the disaster I feared -- it's a bigger one.",
    "Love how the update broke the one feature I actually used.",
    "The instructions weren't unclear, they were just plain wrong.",
    "Can't say I'm surprised the product broke after one use.",
    "Nice, another 'brief' meeting that ran two hours over.",
    "I wasn't happy before, and this didn't help at all.",
]

ambiguous_rows = (
    [{"text": t, "gold_label": "Positive", "category": "ambiguous"} for t in ambiguous_positive]
    + [{"text": t, "gold_label": "Negative", "category": "ambiguous"} for t in ambiguous_negative]
)

# ---- Step 4: combine, shuffle, assign IDs, save ----
all_rows = clear_rows + ambiguous_rows
random.shuffle(all_rows)

for i, row in enumerate(all_rows):
    row["id"] = i + 1

# reorder columns
all_rows = [{"id": r["id"], "text": r["text"], "gold_label": r["gold_label"], "category": r["category"]} for r in all_rows]

with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text", "gold_label", "category"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved {len(all_rows)} sentences to dataset.csv")
print(f"  Clear: {len(clear_rows)} ({len(clear_positive)} pos, {len(clear_negative)} neg)")
print(f"  Ambiguous: {len(ambiguous_rows)} ({len(ambiguous_positive)} pos, {len(ambiguous_negative)} neg)")

# ---- Step 5: pick the 30-sentence subset for the instability check ----
# 15 clear + 15 ambiguous, sampled reproducibly
clear_ids = [r["id"] for r in all_rows if r["category"] == "clear"]
ambiguous_ids = [r["id"] for r in all_rows if r["category"] == "ambiguous"]

random.shuffle(clear_ids)
random.shuffle(ambiguous_ids)

instability_subset_ids = set(clear_ids[:15] + ambiguous_ids[:15])

with open("instability_subset_ids.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id"])
    for i in sorted(instability_subset_ids):
        writer.writerow([i])

print(f"Saved {len(instability_subset_ids)} IDs to instability_subset_ids.csv (for repeat-run stability check)")
