from app.models.schemas import LLMEvidenceRef
from app.validation.citations import locate_quote, quote_is_verbatim, resolve_references


def ref(eid, highlight=None):
    return LLMEvidenceRef(evidence_id=eid, highlight=highlight)


def test_valid_id_resolves_source_timestamp_and_text(app_state):
    res = resolve_references([ref("alpha_02_08")], app_state.store)
    assert not res.dropped
    ev = res.evidence[0]
    assert ev.timestamp == "02:08" and ev.timestamp_seconds == 128
    assert ev.expert_name == "Dr. Clara Alvarez" and ev.market == "Northland"
    raw = (app_state.settings.transcripts_dir / "alpha.txt").read_text(encoding="utf-8")
    assert ev.text in raw  # exact source text


def test_invalid_ids_are_removed(app_state):
    res = resolve_references([ref("alpha_99_99"), ref("../etc/passwd"), ref("alpha_02_08")], app_state.store)
    assert [e.id for e in res.evidence] == ["alpha_02_08"]
    assert len(res.dropped) == 2


def test_ids_outside_retrieved_context_are_removed(app_state):
    res = resolve_references([ref("alpha_02_08")], app_state.store, allowed_ids={"beta_00_18"})
    assert not res.evidence and "retrieved context" in res.dropped[0]


def test_evidence_must_belong_to_the_named_expert(app_state):
    res = resolve_references([ref("beta_00_18")], app_state.store, transcript_id="alpha")
    assert not res.evidence


def test_interviewer_turns_cannot_be_cited(app_state):
    res = resolve_references([ref("alpha_02_00_q")], app_state.store)
    assert not res.evidence


def test_deleted_transcript_evidence_is_rejected(app_state):
    (app_state.settings.transcripts_dir / "beta.md").unlink()
    app_state.refresh()
    assert not resolve_references([ref("beta_00_18")], app_state.store).evidence


def test_verbatim_highlight_is_located(app_state):
    res = resolve_references([ref("alpha_02_08", "the finance team  wants to understand utilisation")], app_state.store)
    hl = res.evidence[0].highlight
    assert hl is not None and hl.text == "The finance team wants to understand utilisation"
    assert res.evidence[0].text[hl.start:hl.end] == hl.text


def test_fabricated_highlight_is_discarded(app_state):
    res = resolve_references([ref("alpha_02_08", "Finance alone decides every purchase")], app_state.store)
    assert res.evidence and res.evidence[0].highlight is None


def test_quote_normalisation():
    src = "It can be 6 to 9 months – but “it can take much longer”."
    assert quote_is_verbatim('it can take much longer', src)
    assert quote_is_verbatim("6 to 9 months - but", src)
    assert not quote_is_verbatim("it always takes 6 months", src)
    assert locate_quote("", src) is None
