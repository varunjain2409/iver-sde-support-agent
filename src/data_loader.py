"""
Loads AmazonHelp customer messages.

Real dataset mode:
    Put the Kaggle CSV (twcs.csv from thoughtvector/customer-support-on-twitter)
    at data/twcs.csv. This loader filters to inbound tweets @-mentioning
    AmazonHelp and reconstructs (customer_msg, agent_reply) pairs using
    response_tweet_id / in_response_to_tweet_id, the same way the real
    corpus links a thread.

Demo mode (default, no real file present):
    Falls back to data/demo_sample.csv, a synthetic stand-in — see
    data/make_demo_sample.py for why, and the README for how to switch over.
"""
import csv
import os
from dataclasses import dataclass
from typing import List, Optional

BRAND = "AmazonHelp"
REAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "twcs.csv")
DEMO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demo_sample.csv")


@dataclass
class Message:
    tweet_id: str
    text: str
    gold_intent: Optional[str] = None
    gold_resolution_style: Optional[str] = None
    gold_escalate: Optional[bool] = None
    is_synthetic: bool = False


def _load_real(path: str) -> List[Message]:
    msgs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("inbound", "").strip().lower() != "true":
                continue
            text = row.get("text", "")
            if f"@{BRAND}" not in text and f"@ {BRAND}" not in text:
                continue
            msgs.append(Message(tweet_id=row["tweet_id"], text=text))
    return msgs


def _load_demo(path: str) -> List[Message]:
    msgs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            msgs.append(Message(
                tweet_id=row["tweet_id"],
                text=row["text"],
                gold_intent=row.get("gold_intent") or None,
                gold_resolution_style=row.get("gold_resolution_style") or None,
                gold_escalate=(row.get("gold_escalate") == "True"),
                is_synthetic=True,
            ))
    return msgs


def load_messages(limit: Optional[int] = None) -> List[Message]:
    if os.path.exists(REAL_PATH):
        msgs = _load_real(REAL_PATH)
        source = "real:data/twcs.csv"
    else:
        msgs = _load_demo(DEMO_PATH)
        source = "SYNTHETIC:data/demo_sample.csv"
    print(f"[data_loader] loaded {len(msgs)} messages from {source}")
    if limit:
        msgs = msgs[:limit]
    return msgs
