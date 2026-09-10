"""
Thin wrapper around an LLM API, used for:
  - llm_classify_intent()   an LLM-based 4th classifier baseline
  - llm_judge_reply()       LLM-as-judge scoring of drafted replies (1-5)

This is OPTIONAL. Nothing else in this repo requires it. If ANTHROPIC_API_KEY
(or OPENAI_API_KEY) is not set, every function here raises RuntimeError with
a clear message, and evaluate.py automatically skips the LLM-judge rows and
says so in its output rather than silently faking a score.
"""
import os
import json

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")


def _require_key():
    if not _ANTHROPIC_KEY and not _OPENAI_KEY:
        raise RuntimeError(
            "No LLM API key found (ANTHROPIC_API_KEY / OPENAI_API_KEY). "
            "LLM-based classification/judging is skipped — see README."
        )


def _call_anthropic(prompt: str, system: str = "") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _call_openai(prompt: str, system: str = "") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_OPENAI_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def _call_llm(prompt: str, system: str = "") -> str:
    _require_key()
    if _ANTHROPIC_KEY:
        return _call_anthropic(prompt, system)
    return _call_openai(prompt, system)


def llm_classify_intent(text: str, intents: list) -> str:
    system = (
        "You classify a customer-support tweet into exactly one intent label "
        f"from this list: {intents}. Reply with ONLY the label, nothing else."
    )
    out = _call_llm(text, system=system).strip().lower()
    return out if out in intents else "general_inquiry"


def llm_judge_reply(customer_msg: str, draft_reply: str) -> dict:
    """
    Returns {"score": 1-5, "reasoning": str} judging the draft reply for:
    relevance, correctness/groundedness, tone, and actionability.
    """
    system = (
        "You are grading a draft customer-support reply. Score 1-5 (5=excellent) "
        "on: does it address the customer's actual issue, is the next step clear, "
        "and is the tone appropriate. Respond ONLY as JSON: "
        '{"score": <int 1-5>, "reasoning": "<one sentence>"}'
    )
    prompt = f"Customer message:\n{customer_msg}\n\nDraft reply:\n{draft_reply}"
    out = _call_llm(prompt, system=system)
    try:
        return json.loads(out)
    except Exception:
        return {"score": None, "reasoning": f"unparsed judge output: {out[:200]}"}
