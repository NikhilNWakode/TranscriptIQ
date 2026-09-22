"""Qualifier-preservation checks.

Research answers are dangerous when a scoped statement ("15–20% in some of the stronger centres")
is flattened into a universal one ("<market> expects 15–20% growth"). This module:

1. finds quantitative claims in a generated answer,
2. locates the cited evidence sentence(s) carrying the same numbers,
3. extracts the qualifiers (scope, condition, hedge, contrast) in those sentences, and
4. flags any qualifier the answer dropped, plus universal wording absent from the evidence.

It is deterministic and provider-independent. In LLM mode, warnings trigger one repair pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..models.schemas import Evidence, QualifierWarning
from ..retrieval.text import fold, split_sentences

_NOUN = r"(?:areas?|centres?|centers?|hospitals?|regions?|sites?|procedures?|institutions?|cases?|trusts?|cities|units?|specialt(?:y|ies)|indications?)"


@dataclass(frozen=True)
class QualifierRule:
    name: str
    detect: re.Pattern[str]
    preserved: Callable[[re.Match[str]], re.Pattern[str]]


def _const(pattern: str) -> Callable[[re.Match[str]], re.Pattern[str]]:
    rx = re.compile(pattern, re.I)
    return lambda _m: rx


def _scope_required(m: re.Match[str]) -> re.Pattern[str]:
    noun = re.search(_NOUN, m.group(0), re.I)
    head = noun.group(0)[:5] if noun else ""
    # must keep the limiting determiner and the scoped noun, e.g. "some ... centres"
    return re.compile(rf"\b(?:some|selected|certain|a few|several|parts of|a number of)\b[^.;]{{0,40}}{re.escape(head)}", re.I)


RULES: list[QualifierRule] = [
    QualifierRule("scope", re.compile(rf"\b(?:in )?(?:some|selected|certain|a few)\b(?: of the)?(?: \w+){{0,2}} {_NOUN}\b", re.I), _scope_required),
    QualifierRule("magnitude", re.compile(r"\b(?:high|low|mid)[- ](?:single|double)[- ]digits?\b", re.I),
                  lambda m: re.compile(re.escape(m.group(0).split()[0].split("-")[0]) + r"[- ](?:single|double)", re.I)),
    QualifierRule("scope", re.compile(r"\bacross the (?:whole|entire) (?:market|country|system)\b", re.I),
                  _const(r"\b(?:whole|entire|overall|nationwide|across the)\b")),
    QualifierRule("condition", re.compile(r"\bif (?:the )?(?:funding|budget|capital|money) (?:is )?(?:already )?(?:available|approved|in place)\b", re.I),
                  _const(r"\b(?:if|when|once|provided|assuming)\b[^.;]{0,40}\b(?:fund|budget|capital|money)")),
    QualifierRule("condition", re.compile(r"\bonce (?:the )?(?:hospital|trust|board|team) (?:becomes|is|gets) (?:serious|committed)\b", re.I),
                  _const(r"\b(?:once|after|when)\b[^.;]{0,40}\b(?:serious|commit)")),
    QualifierRule("condition", re.compile(r"\b(?:depending on|subject to|provided that|unless)\b", re.I),
                  _const(r"\b(?:depend|subject to|provided|unless|if|vary|varies)\b")),
    QualifierRule("hedge", re.compile(r"\bcan take (?:much |a lot |significantly |considerably )?longer\b", re.I),
                  _const(r"\blonger\b")),
    QualifierRule("contrast", re.compile(r"\brather than\b", re.I),
                  _const(r"\b(?:rather than|instead of|not\b[^.;]{0,30}\blike|as opposed to|as against)\b")),
    QualifierRule("scope", re.compile(r"\b(?:larger|bigger|major|big|university|academic|teaching) (?:\w+ )?(?:hospitals|centres|centers|institutions|cities)\b", re.I),
                  _const(r"\b(?:larger|bigger|major|big|university|academic|teaching|leading)\b")),
    QualifierRule("hedge", re.compile(r"\b(?:could|may|might|potentially|possibly|perhaps|up to)\b", re.I),
                  _const(r"\b(?:could|may|might|potential|possibl|perhaps|up to|expect|estimat|anticipat|believ|"
                         r"suggest|think|see|says?|sees?|foresees?|forecasts?|projects?|predicts?)\w*")),
]

UNIVERSAL_RE = re.compile(
    r"\b(?:alone|always|never|universally|everywhere|without exception|sole(?:ly)?|the only factor|"
    r"all (?:hospitals|purchases|decisions|centres|centers)|every (?:hospital|purchase|decision|centre|center)|"
    r"(?:the )?(?:whole|entire) market)\b",
    re.I,
)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_WORD_NUMS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "eighteen": "18",
              "twenty": "20", "thirty": "30"}


_HORIZON_RE = re.compile(r"\b(?:next|over the next|coming|in the next|within)\s+\d+\s*(?:-|–|to)\s*\d+\s*years\b", re.I)


def _numbers(text: str) -> set[str]:
    """Numbers stated in a sentence, ignoring forecast-horizon phrases like "next 3–5 years"."""
    t = _HORIZON_RE.sub(" ", fold(text))
    nums = set(_NUM_RE.findall(t))
    nums |= {v for k, v in _WORD_NUMS.items() if re.search(rf"\b{k}\b", t)}
    return nums


def _is_quantitative_claim(sentence: str) -> bool:
    return bool(re.search(r"\d\s*(?:%|percent|per cent|months?|years?|x\b)|\d+\s*(?:-|–|to)\s*\d+", sentence, re.I))


def _mentions(claim: str, name: str) -> bool:
    """True when the claim names this expert or market (a surname or the market name is enough)."""
    if not name or name == "Unknown":
        return False
    skip = {"the", "and", "dr", "prof", "former", "hospital", "director", "consultant", "head"}
    tokens = [t for t in re.split(r"[\s.,]+", fold(name)) if len(t) > 2 and t not in skip]
    return any(re.search(rf"\b{re.escape(t)}\b", fold(claim)) for t in tokens)


def check_qualifiers(answer: str, evidence: list[Evidence]) -> list[QualifierWarning]:
    warnings: list[QualifierWarning] = []
    seen: set[tuple[str, str]] = set()
    claims = split_sentences(answer) or [answer]
    ev_sentences = [(ev, s) for ev in evidence for s in split_sentences(ev.text)]

    def evidence_for(claim: str) -> list[tuple[Evidence, str]]:
        """A claim that names an expert/market is judged only against that expert's evidence.

        A cross-expert answer states each expert's numbers in its own clause; without this, France's "15-20%"
        matches the UK's "15 percent" evidence and is reported as dropping the UK's qualifiers. Claims that name
        nobody keep the original behaviour of checking against all supplied evidence.
        """
        named = {ev.transcript_id for ev in evidence
                 if _mentions(claim, ev.expert_name) or _mentions(claim, ev.market)}
        return [(ev, s) for ev, s in ev_sentences if ev.transcript_id in named] if named else ev_sentences

    for claim in claims:
        # 1) quantitative claims must keep the qualifiers of the evidence sentence they come from
        if _is_quantitative_claim(claim):
            nums = _numbers(claim)
            for ev, sent in evidence_for(claim):
                shared = nums & _numbers(sent)
                if not shared:
                    continue
                for rule in RULES:
                    for m in rule.detect.finditer(sent):
                        if rule.preserved(m).search(claim):
                            continue
                        key = (claim, m.group(0).lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        warnings.append(QualifierWarning(
                            claim=claim, qualifier=m.group(0), evidence_id=ev.id,
                            message=f"Claim uses {', '.join(sorted(shared))} from {ev.id} but drops the "
                                    f"{rule.name} qualifier “{m.group(0)}”.",
                        ))
        # 2) universal wording that the cited evidence does not itself use
        for m in UNIVERSAL_RE.finditer(claim):
            phrase = m.group(0)
            if any(re.search(rf"\b{re.escape(phrase)}\b", ev.text, re.I) for ev in evidence):
                continue
            key = (claim, phrase.lower())
            if key in seen:
                continue
            seen.add(key)
            warnings.append(QualifierWarning(
                claim=claim, qualifier=phrase, evidence_id=evidence[0].id if evidence else "",
                message=f"Universal wording “{phrase}” is not supported by the cited evidence.",
            ))
    return warnings
