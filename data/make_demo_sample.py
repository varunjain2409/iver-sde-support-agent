"""
Generates data/demo_sample.csv — a SMALL, SYNTHETIC stand-in for the real
Kaggle "Customer Support on Twitter" dataset, filtered to the AmazonHelp brand.

WHY THIS EXISTS
----------------
The real dataset (thoughtvector/customer-support-on-twitter on Kaggle) requires
a Kaggle login to download, and this environment cannot reach kaggle.com or
huggingface.co. This script fabricates ~120 rows in the SAME SCHEMA as the
real dataset so the pipeline below is runnable and testable end-to-end.

BEFORE SUBMITTING FOR REAL: replace data/demo_sample.csv with a genuine
extract of AmazonHelp threads from the real dataset (see README "Using the
real dataset"), and rebuild eval/golden_set.csv by hand-labelling real
examples as the assignment requires. Every result produced from
demo_sample.csv is illustrative only, not a real headline result.
"""
import csv
import random

random.seed(7)

BRAND = "AmazonHelp"

# (customer_text, intent, resolution_style, escalate)
# resolution_style is a short note on how AmazonHelp actually tends to close
# this kind of issue in the real corpus — used to ground the reply drafter.
TEMPLATES = [
    ("Where is my order?? It was supposed to arrive 3 days ago #order{n}",
     "order_status", "ask_for_order_id_then_tracking_link", False),
    ("Package says delivered but I never got it. Order {n}", "delivery_issue",
     "apologize_open_investigation_offer_replacement_or_refund", True),
    ("Item arrived broken, whole thing is cracked. Order {n}", "product_defect",
     "apologize_offer_replacement_or_refund_no_return_needed", True),
    ("I want to return this, how do I start a return for order {n}?",
     "return_refund", "point_to_self_serve_returns_center", False),
    ("Still no refund for order {n} it's been 2 weeks", "return_refund",
     "escalate_to_refund_team_check_bank_processing_time", True),
    ("Can't log into my account, keeps saying wrong password", "account_login",
     "reset_password_link_or_2fa_troubleshooting", False),
    ("Charged twice for order {n}, please fix this", "billing_payment",
     "apologize_open_billing_investigation", True),
    ("What's the status of order {n}?", "order_status",
     "ask_for_order_id_then_tracking_link", False),
    ("The product I got is not what I ordered, wrong item entirely {n}",
     "wrong_item", "apologize_offer_replacement_or_refund_no_return_needed", True),
    ("How do I change my delivery address for order {n}?", "order_status",
     "explain_address_change_window_before_shipping", False),
    ("Your customer service is a joke, third time contacting about {n}",
     "complaint_escalation", "de_escalate_apologize_route_to_specialist", True),
    ("Do you ship to Canada?", "general_inquiry", "provide_shipping_info_link", False),
    ("My gift card isn't working at checkout", "billing_payment",
     "troubleshoot_gift_card_or_escalate_to_billing", True),
    ("Love how fast this arrived, thank you!", "general_inquiry",
     "thank_customer_no_action_needed", False),
    ("Order {n} shows delayed with no new date, it's been a week", "delivery_issue",
     "apologize_open_investigation_offer_replacement_or_refund", True),
]

CLOSERS = [
    "", " pls help", " ??", " this is ridiculous", " thanks", " anyone?", "",
]


def gen_rows(n_threads=200):
    rows = []
    tweet_id = 10_000_000
    for i in range(n_threads):
        cust_txt, intent, resolution, escalate = random.choice(TEMPLATES)
        order_n = random.randint(100000, 999999)
        text = cust_txt.format(n=order_n) + random.choice(CLOSERS)

        cust_id = f"cust_{1000+i}"
        inbound_id = tweet_id
        tweet_id += 1
        response_id = tweet_id
        tweet_id += 1

        rows.append({
            "tweet_id": inbound_id,
            "author_id": cust_id,
            "inbound": True,
            "created_at": f"2019-0{random.randint(1,9)}-{random.randint(10,28):02d} 12:00:00",
            "text": text,
            "response_tweet_id": response_id,
            "in_response_to_tweet_id": "",
            "brand": BRAND,
            "gold_intent": intent,
            "gold_resolution_style": resolution,
            "gold_escalate": escalate,
        })
    return rows


def main():
    rows = gen_rows()
    fields = ["tweet_id", "author_id", "inbound", "created_at", "text",
              "response_tweet_id", "in_response_to_tweet_id", "brand",
              "gold_intent", "gold_resolution_style", "gold_escalate"]
    with open("data/demo_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to data/demo_sample.csv")


if __name__ == "__main__":
    main()
