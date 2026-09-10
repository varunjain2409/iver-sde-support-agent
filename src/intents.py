"""
Intent taxonomy for AmazonHelp, derived by reading ~120 real-style
customer messages and grouping recurring issues (see decision_log.md #1-#3).

INTENTS = {
  "order_status":         tracking / where-is-my-order / address change before ship
  "delivery_issue":       marked delivered but missing, or stuck/delayed in transit
  "return_refund":        wants to return, or refund not received
  "product_defect":       item arrived damaged/broken/not working
  "wrong_item":           received a different item than ordered
  "billing_payment":      double charge, gift card, payment method issue
  "account_login":        can't log in, password, 2FA
  "complaint_escalation": repeat contact, frustrated, demands a human/manager
  "general_inquiry":      shipping policy questions, thanks, other non-issue
}

Three classifiers, increasing in sophistication (used as baselines in eval):
  1. trivial_classifier   — always predicts the majority class. The "do nothing
     smart" baseline every real baseline must beat.
  2. keyword_classifier   — hand-written keyword/regex rules. A simple,
     zero-training baseline.
  3. tfidf_classifier     — TF-IDF + Logistic Regression, trained on the
     golden set itself (small-sample setting — see report "what's misleading").
An LLM-based classifier stub is in llm_backend.py for when an API key is
available; it is NOT required to run the demo pipeline.
"""
import re
from collections import Counter
from typing import List

INTENTS = [
    "order_status", "delivery_issue", "return_refund", "product_defect",
    "wrong_item", "billing_payment", "account_login", "complaint_escalation",
    "general_inquiry",
]

_KEYWORD_RULES = [
    (r"\bpassword\b|\blog ?in\b|\b2fa\b|\bsign ?in\b", "account_login"),
    (r"\bcharged twice\b|\bgift card\b|\bbilling\b|\boverchar", "billing_payment"),
    (r"\bwrong item\b|\bnot what i ordered\b|\bdifferent item\b", "wrong_item"),
    (r"\bbroken\b|\bcracked\b|\bdamaged\b|\bdefect", "product_defect"),
    (r"\brefund\b|\breturn\b", "return_refund"),
    (r"\bdelivered\b.*\bnever got\b|\bdidn'?t (receive|get)\b|\bmissing\b.*\bpackage\b|\bdelayed\b", "delivery_issue"),
    (r"\bthird time\b|\bridiculous\b|\bjoke\b|\bmanager\b|\bterrible service\b", "complaint_escalation"),
    (r"\bwhere is my order\b|\btracking\b|\border status\b|\bchange.*address\b", "order_status"),
    (r"\bship to\b|\bthank you\b|\bthanks\b|\bpolicy\b", "general_inquiry"),
]


def trivial_classifier(texts: List[str], train_labels: List[str] = None):
    """Baseline 1: predict the single most common class every time."""
    majority = Counter(train_labels).most_common(1)[0][0] if train_labels else "order_status"
    return [majority for _ in texts]


def keyword_classifier(texts: List[str]):
    """Baseline 2: first matching regex rule wins; else general_inquiry."""
    preds = []
    for t in texts:
        low = t.lower()
        label = "general_inquiry"
        for pattern, intent in _KEYWORD_RULES:
            if re.search(pattern, low):
                label = intent
                break
        preds.append(label)
    return preds


class TfidfIntentClassifier:
    """
    Baseline 3: TF-IDF + Logistic Regression, fit on labelled examples.
    Deliberately simple — sklearn only, no LLM calls.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, texts: List[str]):
        if not self._fitted:
            raise RuntimeError("call .fit() first")
        X = self.vec.transform(texts)
        return list(self.clf.predict(X))
