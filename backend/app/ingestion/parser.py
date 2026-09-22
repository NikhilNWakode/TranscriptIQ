"""Transcript parser.

Turns raw transcript text into structured, timestamped dialogue segments plus header metadata.
Nothing here is specific to a particular expert, market or transcript: the parser only relies on
generic conventions ("Key: value" headers, "MM:SS Speaker:" dialogue lines).

Design rules
------------
* Source text is preserved verbatim (only whitespace between wrapped lines is normalised and
  wrapping quotation marks are removed).
* Missing metadata is reported as "Unknown" — never guessed from world knowledge.
* Configurable document-conversion artifacts (e.g. a stray "canvas" token) are stripped and
  never become evidence.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

PARSER_VERSION = "1.3"
UNKNOWN = "Unknown"

_TS = r"(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)"
_SPEAKER = r"(?P<speaker>[A-Za-zÀ-ÿ][\wÀ-ÿ.'’\- ]{0,58}?)"

# "01:20 Dr. Martin: text"  |  "[01:20] Dr. Martin - text"  |  "01:20 - Dr. Martin:"
RE_TS_SPEAKER = re.compile(rf"^[\[(]?{_TS}[\])]?\s*[-–—|]?\s*{_SPEAKER}\s*:\s*(?P<text>.*)$")
# "Dr. Martin (01:20): text"  |  "Dr. Martin [01:20] text"
RE_SPEAKER_TS = re.compile(rf"^{_SPEAKER}\s*[\[(]{_TS}[\])]\s*:?\s*(?P<text>.*)$")
# "01:20" alone on a line (speaker on the next line)
RE_TS_ONLY = re.compile(rf"^[\[(]?{_TS}[\])]?\s*$")
# "Dr. Martin: text" (used after a lone timestamp, or for transcripts without timestamps)
RE_SPEAKER_ONLY = re.compile(rf"^{_SPEAKER}\s*:\s*(?P<text>.*)$")

RE_META = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 /]{1,30}?)\s*[:\-–]\s*(?P<value>.*)$")

META_KEYS = {
    "expert_name": {"name", "expert", "expert name", "interviewee", "respondent", "participant"},
    "role": {"role", "title", "job title", "position", "designation", "function"},
    "market": {"market", "country", "region", "geography", "location"},
    "organization": {"organisation", "organization", "company", "hospital", "institution", "employer"},
    "interviewer": {"interviewer", "moderator"},
    "date": {"date", "interview date"},
}
_META_LOOKUP = {alias: key for key, aliases in META_KEYS.items() for alias in aliases}

INTERVIEWER_RE = re.compile(
    r"\b(interviewer|moderator|host|hasamex|analyst|consultant|researcher|question|q)\b", re.I
)
TITLE_RE = re.compile(r"^(dr|prof|professor|mr|mrs|ms|mx|sir|dame)\.?\s+", re.I)


@dataclass
class ParsedSegment:
    seq: int
    speaker_label: str
    speaker: str
    speaker_type: str  # expert | interviewer | other
    timestamp: str | None
    timestamp_seconds: int | None
    text: str
    question_context: str | None


@dataclass
class ParsedTranscript:
    metadata: dict[str, str]
    segments: list[ParsedSegment]
    warnings: list[str] = field(default_factory=list)
    artifacts_removed: int = 0

    @property
    def expert_segments(self) -> list[ParsedSegment]:
        return [s for s in self.segments if s.speaker_type == "expert"]


def timestamp_to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def normalise_timestamp(ts: str) -> str:
    return ":".join(p.zfill(2) for p in ts.split(":"))


def _clean_line(line: str) -> str:
    line = line.replace(" ", " ").replace("﻿", "").rstrip()
    line = re.sub(r"^\s*#{1,6}\s+", "", line)          # markdown headings
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)       # **bold**
    line = re.sub(r"__(.+?)__", r"\1", line)
    line = re.sub(r"^\s*[-*•]\s+(?=\S)", "", line)       # bullets
    return line.strip()


def _strip_artifacts(line: str, tokens: tuple[str, ...]) -> tuple[str, int]:
    """Remove document-conversion artifacts. Returns (cleaned_line, removed_count)."""
    if not tokens:
        return line, 0
    alt = "|".join(re.escape(t) for t in tokens)
    if line.strip().lower() in tokens:
        return "", 1
    removed = 0
    # (pattern, replacement, flags) — each only fires at a structural position. Rules that touch running text are
    # case-sensitive and require the artifact to be followed by a capitalised word or quote ("canvas The ..."),
    # which a genuine sentence would not produce ("... on canvas bags ...").
    rules = [
        (rf"^(?:{alt})\s+(?=[\[(]?\d{{1,2}}:\d{{2}})", "", re.I),                              # before a timestamp
        (rf"^([\[(]?\d{{1,2}}:\d{{2}}(?::\d{{2}})?[\])]?)\s+(?:{alt})\b\s*", r"\1 ", re.I),      # right after a timestamp
        (rf":\s*(?:{alt})\s*$", ":", re.I),                                                     # "Speaker: canvas"
        (rf"^([^:\"“]{{1,60}}:)\s*(?:{alt})\s+(?=[A-Z\"“])", r"\1 ", 0),                         # "Speaker: canvas The ..."
        (rf"([.!?\"”'’])\s+(?:{alt})\s*$", r"\1", re.I),                                        # trailing after sentence end
        (rf"([.!?\"”])\s+(?:{alt})\s+(?=[A-Z\"“])", r"\1 ", 0),                                  # between two sentences
        (rf"^(?:{alt})\s+(?=[A-Z\"“])", "", 0),                                                 # start of a text line
    ]
    for pattern, repl, flags in rules:
        new = re.sub(pattern, repl, line, flags=flags)
        if new != line:
            removed += 1
            line = new
    return line.strip(), removed


def _strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    pairs = [('"', '"'), ("“", "”"), ("„", "“"), ("«", "»")]
    for open_q, close_q in pairs:
        if len(t) >= 2 and t.startswith(open_q) and t.endswith(close_q):
            inner = t[1:-1]
            # only strip when the quotes wrap the whole utterance
            if open_q not in inner and close_q not in inner:
                return inner.strip()
    # utterance opened with a quote whose closing quote was lost (or vice versa)
    if t[:1] in {'"', "“"} and t.count('"') + t.count("“") + t.count("”") == 1:
        return t[1:].strip()
    if t[-1:] in {'"', "”"} and t.count('"') + t.count("“") + t.count("”") == 1:
        return t[:-1].strip()
    return t


def _valid_speaker(label: str) -> bool:
    label = label.strip()
    if not label or len(label.split()) > 6:
        return False
    if label.lower() in _META_LOOKUP and label.lower() not in {"interviewer", "moderator", "expert"}:
        return False
    return bool(re.match(r"^[A-Za-zÀ-ÿ]", label))


def _name_tokens(name: str) -> list[str]:
    name = TITLE_RE.sub("", name.strip())
    return [t.lower().strip(".,") for t in re.split(r"\s+", name) if t.strip(".,")]


def speaker_matches_name(label: str, full_name: str) -> bool:
    """'Dr. Smith' matches 'Dr. Jane Smith'; 'Jane' matches 'Jane Smith'."""
    if not full_name or full_name == UNKNOWN:
        return False
    lt, nt = _name_tokens(label), _name_tokens(full_name)
    if not lt or not nt:
        return False
    return set(lt).issubset(set(nt)) or lt[-1] == nt[-1]


def _parse_metadata(lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = RE_META.match(line)
        if m:
            key = _META_LOOKUP.get(re.sub(r"\s*\d+$", "", m.group("key").strip().lower()))
            if key and key not in meta:
                value = m.group("value").strip()
                if not value:
                    # "Name:" on one line, value on the next non-empty line
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and not RE_META.match(lines[j]):
                        value = lines[j].strip()
                        i = j
                value = value.strip(" \"'“”")
                if value:
                    meta[key] = value
        i += 1
    return meta


@dataclass
class _RawTurn:
    speaker_label: str
    timestamp: str | None
    parts: list[str]


def _is_dialogue_start(line: str) -> bool:
    for rx in (RE_TS_SPEAKER, RE_SPEAKER_TS):
        m = rx.match(line)
        if m and _valid_speaker(m.group("speaker")):
            return True
    return bool(RE_TS_ONLY.match(line))


def parse_transcript(raw_text: str, artifact_tokens: tuple[str, ...] = ("canvas",)) -> ParsedTranscript:
    warnings: list[str] = []
    artifacts = 0
    lines: list[str] = []
    for raw_line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned, removed = _strip_artifacts(_clean_line(raw_line), artifact_tokens)
        artifacts += removed
        lines.append(cleaned)

    has_timestamps = any(_is_dialogue_start(line) for line in lines)
    first_dialogue = next((i for i, line in enumerate(lines) if _is_dialogue_start(line)), len(lines))
    if not has_timestamps:
        first_dialogue = next(
            (i for i, line in enumerate(lines)
             if (m := RE_SPEAKER_ONLY.match(line)) and _valid_speaker(m.group("speaker"))
             and m.group("speaker").strip().lower() not in _META_LOOKUP),
            len(lines),
        )

    metadata = _parse_metadata(lines[:first_dialogue])

    turns: list[_RawTurn] = []
    pending_ts: str | None = None
    for line in lines[first_dialogue:]:
        if not line:
            continue
        m = RE_TS_SPEAKER.match(line) or RE_SPEAKER_TS.match(line)
        if m and _valid_speaker(m.group("speaker")):
            turns.append(_RawTurn(m.group("speaker").strip(), m.group("ts"), [m.group("text")]))
            pending_ts = None
            continue
        m = RE_TS_ONLY.match(line)
        if m:
            pending_ts = m.group("ts")
            continue
        m = RE_SPEAKER_ONLY.match(line)
        if m and _valid_speaker(m.group("speaker")) and (pending_ts or not has_timestamps):
            turns.append(_RawTurn(m.group("speaker").strip(), pending_ts, [m.group("text")]))
            pending_ts = None
            continue
        if turns:
            turns[-1].parts.append(line)

    turns = [t for t in turns if " ".join(p for p in t.parts if p).strip()]

    if not turns:
        warnings.append("No speaker-labelled dialogue found; paragraphs were indexed without timestamps.")
        body = "\n".join(lines[first_dialogue:] if first_dialogue < len(lines) else lines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        name = metadata.get("expert_name", UNKNOWN)
        segs = [
            ParsedSegment(i, name, name, "expert", None, None, " ".join(p.split()), None)
            for i, p in enumerate(paragraphs)
            if not RE_META.match(p) or len(p) > 80
        ]
        return ParsedTranscript(_finalise_metadata(metadata), segs, warnings, artifacts)

    # --- speaker classification ---------------------------------------------------------
    labels = Counter(t.speaker_label for t in turns)
    interviewer_name = metadata.get("interviewer", "")
    expert_name = metadata.get("expert_name", "")

    def is_interviewer(label: str) -> bool:
        if interviewer_name and speaker_matches_name(label, interviewer_name):
            return True
        if expert_name and speaker_matches_name(label, expert_name):
            return False
        return bool(INTERVIEWER_RE.search(label))

    interviewer_labels = {lab for lab in labels if is_interviewer(lab)}
    if not interviewer_labels and len(labels) == 2:
        # fall back to "who asks the questions"
        def q_ratio(label: str) -> float:
            own = [" ".join(t.parts) for t in turns if t.speaker_label == label]
            return sum(1 for x in own if x.rstrip(" \"”").endswith("?")) / max(len(own), 1)

        interviewer_labels = {max(labels, key=q_ratio)}

    candidates = [lab for lab in labels if lab not in interviewer_labels]
    if expert_name:
        expert_labels = {lab for lab in candidates if speaker_matches_name(lab, expert_name)}
        if not expert_labels and len(candidates) == 1:
            expert_labels = set(candidates)
            warnings.append(f"Expert speaker label '{candidates[0]}' did not match header name; assumed expert.")
    else:
        expert_labels = {max(candidates, key=lambda lab: labels[lab])} if candidates else set()
        if expert_labels:
            metadata["expert_name"] = next(iter(expert_labels))
            warnings.append("Expert name not found in header; using the most frequent non-interviewer speaker.")
    if not interviewer_labels:
        warnings.append("No interviewer detected; question context unavailable.")

    # --- build segments ------------------------------------------------------------------
    resolved_expert = metadata.get("expert_name", UNKNOWN)
    segments: list[ParsedSegment] = []
    last_question: str | None = None
    for seq, turn in enumerate(turns):
        text = _strip_wrapping_quotes(" ".join(" ".join(p for p in turn.parts if p).split()))
        if turn.speaker_label in interviewer_labels:
            stype, speaker = "interviewer", turn.speaker_label
        elif turn.speaker_label in expert_labels:
            stype, speaker = "expert", resolved_expert
        else:
            stype, speaker = "other", turn.speaker_label
        ts = normalise_timestamp(turn.timestamp) if turn.timestamp else None
        segments.append(
            ParsedSegment(
                seq=seq,
                speaker_label=turn.speaker_label,
                speaker=speaker,
                speaker_type=stype,
                timestamp=ts,
                timestamp_seconds=timestamp_to_seconds(ts) if ts else None,
                text=text,
                question_context=last_question if stype != "interviewer" else None,
            )
        )
        if stype == "interviewer":
            last_question = text

    if not any(s.speaker_type == "expert" for s in segments):
        warnings.append("No expert statements detected.")
    return ParsedTranscript(_finalise_metadata(metadata), segments, warnings, artifacts)


def _finalise_metadata(meta: dict[str, str]) -> dict[str, str]:
    return {
        "expert_name": meta.get("expert_name") or UNKNOWN,
        "role": meta.get("role") or UNKNOWN,
        "market": meta.get("market") or UNKNOWN,
        **{k: v for k, v in meta.items() if k in {"organization", "interviewer", "date"}},
    }
