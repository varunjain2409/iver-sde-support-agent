"""
Runnable end-to-end pipeline: load messages -> classify intent -> draft
reply -> decide escalation -> write results CSV.

Usage:
    python -m src.pipeline --limit 50 --out outputs/pipeline_results.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_messages
from src.intents import keyword_classifier
from src.reply_generation import draft_reply
from src.escalation import decide_escalation


def run(limit=None, out_path="outputs/pipeline_results.csv", classifier="keyword"):
    messages = load_messages(limit=limit)
    texts = [m.text for m in messages]

    if classifier == "keyword":
        preds = keyword_classifier(texts)
    elif classifier == "tfidf":
        from src.intents import TfidfIntentClassifier
        gold_available = [m for m in messages if m.gold_intent]
        if len(gold_available) < 5:
            raise RuntimeError("tfidf classifier needs labelled training data; use --classifier keyword, or provide a golden set to train on")
        clf = TfidfIntentClassifier().fit([m.text for m in gold_available], [m.gold_intent for m in gold_available])
        preds = clf.predict(texts)
    else:
        raise ValueError(f"unknown classifier {classifier}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "text", "predicted_intent", "draft_reply", "escalate", "escalation_reason"])
        for m, intent in zip(messages, preds):
            reply = draft_reply(intent, m.text)
            dec = decide_escalation(intent, m.text)
            w.writerow([m.tweet_id, m.text, intent, reply, dec.escalate, dec.reason])

    print(f"wrote {len(messages)} rows to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=str, default="outputs/pipeline_results.csv")
    p.add_argument("--classifier", type=str, default="keyword", choices=["keyword", "tfidf"])
    args = p.parse_args()
    run(limit=args.limit, out_path=args.out, classifier=args.classifier)
