# Decision log

1. **Brand: AmazonHelp.** Highest tweet volume in the real dataset and the
   widest spread of issue types (order, delivery, refunds, billing, login),
   which makes for a more interesting 9-intent taxonomy than a narrower brand.

2. **9 intents, not more.** Early draft had 14 intents (split delivery issues
   into "late" vs "lost" vs "wrong address", etc.). Collapsed to 9 because
   several splits didn't change the reply or escalation decision — if two
   intents get the same treatment downstream, they're not worth
   distinguishing for this agent's purpose.

3. **`complaint_escalation` is its own intent, not a modifier on other
   intents.** A frustrated repeat-contact about a late order is functionally
   different from a first-time late-order question, even though the root
   cause is the same "order_status"-shaped complaint. Treating frustration
   as its own top-level intent made the escalation rule trivial to write.

4. **Reply grounding is intent-level, not per-tweet retrieval.** Considered
   retrieving the single most-similar historical resolved tweet per new
   message (classic RAG). Rejected for this data volume: with only ~200
   threads, per-tweet retrieval would often retrieve a poor match; a
   hand-curated per-intent resolution summary is more reliable at small
   scale, at the cost of losing tweet-specific nuance.

5. **Escalation is a rule set keyed on intent, not a learned classifier.**
   Money/replacement decisions need to be auditable — "why did the bot
   escalate this" needs a one-line human-readable answer, which a small
   rule set gives for free and a learned model doesn't.

6. **`return_refund` is the one intent with a *conditional* escalation
   rule** (overdue language triggers escalate, otherwise auto-handle),
   while every other intent's escalation is a flat yes/no. This is because
   it's the one intent where the *same* customer action (asking about a
   return/refund) has two very different underlying situations — normal
   process question vs. something already gone wrong.

7. **Defaulting unrecognized-intent to escalate, not auto-handle.** If the
   classifier ever returns something outside the known 9 intents (shouldn't
   happen with the current classifiers, but matters if this is swapped for
   an LLM classifier later), fail toward a human rather than toward
   auto-replying to something unclassified.

8. **Trivial baseline predicts training-set majority class, not a fixed
   guess like "general_inquiry".** Makes it a fair "does the smarter model
   beat a model with zero domain knowledge" comparison rather than a
   strawman.

9. **Chose not to build a learned escalation classifier at all** (see
   README "chose not to build") — with ~180 labelled examples split across
   9 intents, there usually aren't enough escalate/don't-escalate examples
   per intent to train something more reliable than the rule set, and the
   rule set is easier to defend live.

10. **`llm_backend.py` fails loudly (`RuntimeError`) rather than silently
    falling back to a fake score when no API key is present.** Evaluation
    numbers should never look complete when a component didn't actually
    run.

11. **70/30 train/test split for the TF-IDF baseline inside `evaluate.py`**,
    rather than training and testing on the full golden set. Otherwise the
    "simple baseline" number would just be measuring memorization of the
    exact 180 examples, which isn't a meaningful signal even as a baseline.

12. **Synthetic demo data uses exactly 15 fixed templates, not paraphrased
    variations.** This was deliberate, not laziness — it makes the "what's
    misleading about my headline number" analysis honest and checkable: a
    reviewer can see precisely why 1.00 accuracy is hollow, instead of me
    asserting it.

13. **Kept the reply drafter template-based rather than calling an LLM by
    default**, even though an LLM path exists. Free-generation risks
    inventing policy specifics (e.g. return windows) not grounded in any
    real historical resolution — a wrong templated reply is at least a
    known, auditable failure mode; a wrong generated reply looks equally
    confident and is harder to catch.

14. **Escalation reason strings are short and cite the intent/pattern, not
    a verbatim quote of the trigger phrase** (flagged as failure mode #3 in
    the README) — a deliberate v1 simplification given time, not an
    oversight; documented as the first thing to fix with another week.
