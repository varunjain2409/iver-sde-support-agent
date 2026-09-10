"""
Decides auto-handle vs escalate-to-human, with a stated reason.

Rule (see decision_log.md #5-#7 for why these specific triggers):
  - complaint_escalation intent            -> always escalate (frustration /
                                               repeat-contact signal)
  - delivery_issue, product_defect,
    wrong_item, billing_payment            -> escalate (money/replacement
                                               decision a bot shouldn't make
                                               unilaterally)
  - return_refund                          -> escalate ONLY if the message
                                               implies refund is overdue
                                               ("still", "weeks", "no refund"),
                                               else auto-handle (self-serve
                                               return flow is enough)
  - order_status, account_login,
    general_inquiry                       -> auto-handle
"""
import re
from dataclasses import dataclass

ALWAYS_ESCALATE = {"complaint_escalation", "delivery_issue", "product_defect",
                    "wrong_item", "billing_payment"}
NEVER_ESCALATE = {"order_status", "account_login", "general_inquiry"}
_OVERDUE_PAT = re.compile(r"\bstill\b|\bweeks?\b|\bno refund\b|\bhaven'?t (received|gotten)\b")


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def decide_escalation(intent: str, customer_text: str) -> EscalationDecision:
    if intent in ALWAYS_ESCALATE:
        return EscalationDecision(True, f"intent '{intent}' involves money/replacement or visible frustration — routed to a human")
    if intent == "return_refund":
        if _OVERDUE_PAT.search(customer_text.lower()):
            return EscalationDecision(True, "refund appears overdue ('still'/'weeks'/'no refund' language) — routed to a human to check processing")
        return EscalationDecision(False, "standard return request — self-serve return flow is sufficient")
    if intent in NEVER_ESCALATE:
        return EscalationDecision(False, f"intent '{intent}' is informational/self-serve — auto-handled")
    return EscalationDecision(True, "unrecognized intent pattern — defaulting to human review")
