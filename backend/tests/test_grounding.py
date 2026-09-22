"""End-to-end grounding: LLM output is reduced to what the backend can verify."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import NO_EVIDENCE_MESSAGE, AskRequest


def ask(state, q, **filters):
    from app.models.schemas import AskFilters

    return state.research.ask(AskRequest(question=q, filters=AskFilters(**filters)))


# ---- extractive (mock) mode ----------------------------------------------------------------
def test_supported_question_returns_source_evidence(app_state):
    r = ask(app_state, "Which expert discusses training capacity as a barrier?")
    assert not r.insufficient_evidence
    assert "beta_00_18" in [s.id for s in r.sources]
    src = next(s for s in r.sources if s.id == "beta_00_18")
    assert src.timestamp == "00:18" and src.expert_name == "Jonas Brandt" and src.market == "Southland"


def test_unsupported_question_returns_no_evidence(app_state):
    r = ask(app_state, "What is the weather forecast in Tokyo next week?")
    assert r.insufficient_evidence and r.answer == NO_EVIDENCE_MESSAGE and r.sources == []


def test_filters_restrict_experts(app_state):
    r = ask(app_state, "What is the typical purchasing timeline?", markets=["Northland"])
    assert {s.transcript_id for s in r.sources} == {"alpha"}


def test_every_guide_question_answers_every_expert(app_state):
    _, questions = app_state.research.guide_questions()
    experts = app_state.store.experts()
    for q in questions:
        res = app_state.research.answer_guide_question(q)
        assert len(res.expert_answers) == len(experts)
        for a in res.expert_answers:
            assert a.evidence or not a.supported
            for e in a.evidence:
                assert e.transcript_id == a.transcript_id


# ---- LLM mode with a fake provider -----------------------------------------------------------
def test_llm_citations_are_resolved_and_fabrications_dropped(app_state, with_fake_llm):
    with_fake_llm(lambda user, schema: {
        "insufficient_evidence": False,
        "answer": "Jonas Brandt identifies training capacity as a bottleneck.",
        "expert_answers": [
            {"transcript_id": "beta", "answer": "Training capacity is a bottleneck.",
             "evidence": [{"evidence_id": "beta_00_18", "highlight": "not have enough proctors"},
                          {"evidence_id": "alpha_02_08", "highlight": None}]},   # wrong expert
            {"transcript_id": "ghost", "answer": "Invented expert.", "evidence": []},
        ],
        "evidence": [{"evidence_id": "beta_00_18", "highlight": None},
                     {"evidence_id": "beta_12_34", "highlight": None}],          # does not exist
    })
    r = ask(app_state, "Which expert discusses training capacity as a barrier?")
    assert [s.id for s in r.sources] == ["beta_00_18"]
    assert r.sources[0].text.startswith("Training capacity is a real bottleneck.")
    assert r.expert_answers[0].evidence[0].highlight.text == "not have enough proctors"
    assert len(r.expert_answers) == 1
    assert len(r.meta.dropped_citations) >= 3


def test_llm_with_only_invalid_citations_yields_no_evidence(app_state, with_fake_llm):
    with_fake_llm(lambda user, schema: {
        "insufficient_evidence": False, "answer": "Hospitals always buy robots.",
        "expert_answers": [], "evidence": [{"evidence_id": "made_up_01_00", "highlight": None}],
    })
    r = ask(app_state, "What are the barriers to adoption?")
    assert r.insufficient_evidence and r.answer == NO_EVIDENCE_MESSAGE and not r.sources


def test_llm_insufficient_flag_is_respected(app_state, with_fake_llm):
    with_fake_llm(lambda user, schema: {"insufficient_evidence": True, "answer": "n/a", "expert_answers": [],
                                        "evidence": [{"evidence_id": "alpha_02_08", "highlight": None}]})
    r = ask(app_state, "What is the budget for marketing campaigns?")
    assert r.insufficient_evidence and r.sources == []


def test_qualifier_warning_triggers_repair(app_state, with_fake_llm):
    calls = []

    def responder(user, schema):
        calls.append(user)
        answer = ("Dr. Alvarez expects 15 to 20% more procedures each year in some of the stronger centres."
                  if "validator reviewed" in user else "Northland expects 15–20% growth.")
        return {"insufficient_evidence": False, "answer": answer, "expert_answers": [],
                "evidence": [{"evidence_id": "alpha_03_18", "highlight": None}]}

    with_fake_llm(responder)
    r = ask(app_state, "What adoption trend do experts expect?")
    assert len(calls) == 2
    assert "some of the stronger centres" in r.answer and r.qualifier_warnings == []


def test_llm_disagreement_needs_two_experts(app_state, with_fake_llm):
    def responder(user, schema):
        if schema.__name__ == "LLMFindings":
            return {"findings": [
                {"title": "Growth outlook", "summary": "Dr. Alvarez expects growth in some of the stronger centres.",
                 "classification": "disagreement", "transcript_ids": ["alpha", "beta"],
                 "evidence": [{"evidence_id": "alpha_03_18", "highlight": None}]},
                {"title": "Invented theme", "summary": "Unsupported.", "classification": "common_view",
                 "transcript_ids": ["alpha"], "evidence": [{"evidence_id": "nope_00_00", "highlight": None}]},
            ]}
        raise AssertionError(schema)

    with_fake_llm(responder)
    r = app_state.research.insights()
    assert r.disagreements == []  # single-expert "disagreement" downgraded
    assert all(f.title != "Invented theme" for f in r.by_question + r.themes)
    assert all(f.evidence for f in r.by_question + r.themes)
    assert r.experts_analyzed == 2 and r.markets_represented == 2


# ---- HTTP API --------------------------------------------------------------------------------
def test_api_endpoints(app_state):
    app = create_app(app_state, sync_on_startup=False)
    with TestClient(app) as client:
        assert client.get("/api/stats").json()["transcripts"] == 2
        assert client.get("/api/evidence/alpha_02_08").json()["timestamp"] == "02:08"
        assert client.get("/api/evidence/alpha_99_99").status_code == 404
        assert client.get("/api/evidence/..%2Fsecret").status_code in (400, 404)
        assert client.post("/api/ask", json={"question": ""}).status_code == 422
        r = client.post("/api/ask", json={"question": "What is the typical purchasing timeline?"}).json()
        assert r["sources"]
        guide = client.post("/api/interview-guide/answer", json={"question_id": "q6"}).json()
        assert len(guide) == 1 and len(guide[0]["expert_answers"]) == 2
        assert set(client.get("/api/filters").json()["markets"]) == {"Northland", "Southland"}
        detail = client.get("/api/transcripts/alpha").json()
        assert detail["segments_list"][0]["timestamp"] == "00:15"
        assert client.post("/api/refresh").json()["unchanged"] == ["alpha", "beta"]
        assert client.get("/api/insights").status_code == 200


def test_generic_domain_words_do_not_satisfy_evidence_gate(app_state):
    # "robotic surgery" is everywhere in the corpus; the specific subject ("company", "sells") is nowhere.
    for q in ["Which company sells the most robotic surgery systems?",
              "What is the average price of a robotic surgery system?"]:
        r = ask(app_state, q)
        assert r.insufficient_evidence and r.answer == NO_EVIDENCE_MESSAGE, q


def test_idf_weighted_coverage(app_state):
    ret = app_state.retriever
    q_ok, q_bad = "How important is ROI?", "Which company sells the most robotic surgery systems?"
    assert ret.term_coverage(q_ok, ret.retrieve(q_ok)) == 1.0
    assert ret.term_coverage(q_bad, ret.retrieve(q_bad)) < 0.75


def test_extractive_answer_leads_with_strongest_match(app_state):
    r = ask(app_state, "Which expert discusses training capacity as a barrier?")
    assert r.answer.startswith("Strongest match — Jonas Brandt (Southland) at 00:18")
