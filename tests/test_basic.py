import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.intents import keyword_classifier, trivial_classifier
from src.escalation import decide_escalation
from src.reply_generation import draft_reply


def test_keyword_classifier_basic_cases():
    texts = [
        "Where is my order? #order123",
        "Can't log into my account, wrong password",
        "Item arrived broken and cracked",
    ]
    preds = keyword_classifier(texts)
    assert preds[0] == "order_status"
    assert preds[1] == "account_login"
    assert preds[2] == "product_defect"


def test_trivial_classifier_returns_majority():
    labels = ["a", "a", "a", "b"]
    preds = trivial_classifier(["x", "y"], train_labels=labels)
    assert preds == ["a", "a"]


def test_escalation_always_escalates_product_defect():
    dec = decide_escalation("product_defect", "it's broken")
    assert dec.escalate is True
    assert "reason" not in dec.reason  # sanity: reason is a real sentence, not the literal word


def test_escalation_return_refund_overdue_vs_not():
    overdue = decide_escalation("return_refund", "still no refund after 2 weeks")
    fresh = decide_escalation("return_refund", "how do I start a return?")
    assert overdue.escalate is True
    assert fresh.escalate is False


def test_draft_reply_nonempty_for_all_known_intents():
    for intent in ["order_status", "delivery_issue", "return_refund", "product_defect",
                   "wrong_item", "billing_payment", "account_login",
                   "complaint_escalation", "general_inquiry"]:
        reply = draft_reply(intent)
        assert isinstance(reply, str) and len(reply) > 10
