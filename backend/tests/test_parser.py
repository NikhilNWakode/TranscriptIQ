from conftest import FIXTURES

from app.ingestion.indexer import evidence_ids
from app.ingestion.parser import UNKNOWN, parse_transcript, speaker_matches_name, timestamp_to_seconds


def parse(name: str):
    return parse_transcript((FIXTURES / name).read_text(encoding="utf-8"))


def test_metadata_value_on_next_line():
    p = parse("alpha.txt")
    assert p.metadata["expert_name"] == "Dr. Clara Alvarez"
    assert p.metadata["role"] == "Head of Surgery"
    assert p.metadata["market"] == "Northland"


def test_metadata_markdown_inline():
    p = parse("beta.md")
    assert p.metadata == {"expert_name": "Jonas Brandt", "role": "Procurement Director", "market": "Southland"}


def test_timestamps_extracted_and_converted():
    p = parse("alpha.txt")
    assert [s.timestamp for s in p.segments][:4] == ["00:15", "00:22", "01:05", "01:12"]
    seg = next(s for s in p.segments if s.timestamp == "02:08")
    assert seg.timestamp_seconds == 128
    assert timestamp_to_seconds("01:02:03") == 3723


def test_speakers_interviewer_and_expert_detection():
    p = parse("alpha.txt")
    types = {s.timestamp: s.speaker_type for s in p.segments}
    assert types["00:15"] == "interviewer" and types["00:22"] == "expert"
    # the short label "Dr. Alvarez" resolves to the full header name
    expert = next(s for s in p.segments if s.timestamp == "00:22")
    assert expert.speaker_label == "Dr. Alvarez" and expert.speaker == "Dr. Clara Alvarez"


def test_mixed_short_and_full_speaker_labels():
    p = parse("beta.md")
    assert {s.speaker for s in p.expert_segments} == {"Jonas Brandt"}
    assert len(p.expert_segments) == 4


def test_question_context_association():
    p = parse("alpha.txt")
    seg = next(s for s in p.segments if s.timestamp == "01:12")
    assert seg.question_context == "What is holding adoption back?"
    assert next(s for s in p.segments if s.timestamp == "00:15").question_context is None


def test_exact_text_preserved_and_quotes_unwrapped():
    p = parse("alpha.txt")
    seg = next(s for s in p.segments if s.timestamp == "02:08")
    assert seg.text == ("Very important. The finance team wants to understand utilisation, procedure volume and "
                        "maintenance cost before anything is signed.")


def test_canvas_artifacts_never_become_evidence():
    for name in ("alpha.txt", "beta.md"):
        p = parse(name)
        assert p.artifacts_removed >= 1
        assert all("canvas" not in s.text.lower() for s in p.segments)
        assert all(s.text.strip() for s in p.segments)


def test_inline_artifacts_around_timestamps():
    raw = "Name: A B\n\ncanvas 00:10 Interviewer: Why?\n00:20 canvas A B: Because it works.\n00:30 A B: Fine. canvas\n"
    p = parse_transcript(raw)
    assert [s.text for s in p.segments] == ["Why?", "Because it works.", "Fine."]


def test_artifact_word_inside_sentence_is_kept():
    raw = "00:10 Interviewer: Q?\n00:20 Pat Lee: We paint on canvas bags in the workshop.\n"
    p = parse_transcript(raw)
    assert p.segments[1].text == "We paint on canvas bags in the workshop."


def test_missing_metadata_is_unknown_not_invented():
    raw = "00:05 Interviewer: What do you think?\n00:12 Sam Roe: It depends on funding.\n"
    p = parse_transcript(raw)
    assert p.metadata["role"] == UNKNOWN and p.metadata["market"] == UNKNOWN
    assert p.metadata["expert_name"] == "Sam Roe"  # most frequent non-interviewer speaker, flagged in warnings
    assert any("Expert name not found" in w for w in p.warnings)


def test_speaker_timestamp_layout_and_hhmmss():
    raw = "Expert: Kim Park\nMarket: Eastmark\n\nModerator (00:00:05): Hello?\nKim Park (00:01:10): Growth is slow.\n"
    p = parse_transcript(raw)
    assert p.segments[1].timestamp == "00:01:10" and p.segments[1].timestamp_seconds == 70
    assert p.segments[0].speaker_type == "interviewer"


def test_interviewer_inferred_from_questions_when_unlabelled():
    raw = "00:01 Ana: Why is it slow?\n00:05 Ben: Budgets.\n00:09 Ana: And training?\n00:12 Ben: Also.\n"
    p = parse_transcript(raw)
    assert {s.speaker_label for s in p.segments if s.speaker_type == "interviewer"} == {"Ana"}


def test_malformed_transcript_falls_back_to_paragraphs():
    p = parse_transcript("Some notes without any speakers.\n\nA second paragraph.")
    assert len(p.segments) == 2 and p.warnings


def test_evidence_ids_are_deterministic_and_unique():
    p = parse("alpha.txt")
    ids = evidence_ids("alpha", p)
    assert ids == evidence_ids("alpha", parse("alpha.txt"))
    assert "alpha_02_08" in ids and "alpha_00_15_q" in ids
    assert len(set(ids)) == len(ids)


def test_speaker_name_matching():
    assert speaker_matches_name("Dr. Alvarez", "Dr. Clara Alvarez")
    assert speaker_matches_name("Clara", "Dr. Clara Alvarez")
    assert not speaker_matches_name("Interviewer", "Dr. Clara Alvarez")


def test_artifact_after_speaker_label_and_between_sentences():
    raw = ("Name: Pat Lee\n\n00:10\nInterviewer: Why?\n\n00:12\nPat Lee: canvas The main barrier is nurses.\n\n"
           "00:20\nPat Lee: Four months. canvas It varies.\ncanvas \"Quoted continuation.\"\n")
    p = parse_transcript(raw)
    assert [s.text for s in p.expert_segments] == ["The main barrier is nurses.",
                                                   'Four months. It varies. "Quoted continuation."']
    assert p.artifacts_removed == 3


def test_timestamp_on_own_line_and_numbered_expert_header():
    raw = "Expert 1 – Dr. Jane Smith\nRole: Surgeon\nMarket: Eastmark\n\n00:00\nInterviewer: Hi?\n\n00:18\nDr. Smith: Hello.\n"
    p = parse_transcript(raw)
    assert p.metadata == {"expert_name": "Dr. Jane Smith", "role": "Surgeon", "market": "Eastmark"}
    assert (p.segments[1].timestamp, p.segments[1].speaker, p.segments[1].question_context) == ("00:18", "Dr. Jane Smith", "Hi?")
