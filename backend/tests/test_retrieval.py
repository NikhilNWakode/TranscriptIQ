from app.models.schemas import AskFilters


def test_relevant_evidence_is_retrieved(app_state):
    hits = app_state.retriever.retrieve("How important is ROI for the finance team?", per_expert_k=1)
    top_alpha = next(h for h in hits if h.transcript_id == "alpha")
    assert top_alpha.evidence_id == "alpha_02_08"


def test_training_capacity_found(app_state):
    hits = app_state.retriever.retrieve("Which expert discusses training capacity as a barrier?")
    assert hits[0].evidence_id == "beta_00_18"


def test_cross_expert_retrieval_is_balanced(app_state):
    hits = app_state.retriever.retrieve("What are the typical purchasing decision timelines in months?", per_expert_k=2)
    assert {h.transcript_id for h in hits} == {"alpha", "beta"}
    assert sum(1 for h in hits if h.transcript_id == "alpha") <= 2


def test_metadata_filtering_by_market(app_state):
    hits = app_state.retriever.retrieve("timeline", filters=AskFilters(markets=["Southland"]))
    assert hits and {h.transcript_id for h in hits} == {"beta"}


def test_metadata_filtering_by_transcript(app_state):
    hits = app_state.retriever.retrieve("budget", filters=AskFilters(transcript_ids=["alpha"]))
    assert hits and all(h.transcript_id == "alpha" for h in hits)


def test_only_expert_statements_are_retrievable(app_state):
    hits = app_state.retriever.retrieve("How would you describe current adoption?", per_expert_k=10, max_total=50)
    assert all(not h.evidence_id.endswith("_q") for h in hits)


def test_term_coverage(app_state):
    hits = app_state.retriever.retrieve("weather forecast in Tokyo")
    assert app_state.retriever.term_coverage("weather forecast in Tokyo", hits) == 0


def test_query_expansion_matches_paraphrased_interviewer_question(app_state):
    # "main barriers" vs the interviewer's "What is holding adoption back?" — no shared subject word
    hits = app_state.retriever.retrieve_for_expert("What are the main barriers to adoption?", "alpha", k=1)
    assert hits[0].evidence_id == "alpha_01_12"


def test_query_expander_is_word_bounded_and_config_driven():
    from app.retrieval.retriever import QueryExpander

    qe = QueryExpander([["roi", "pay for itself"], ["use", "usage"]])
    assert "pay for itself" in qe.expand("How important is ROI?")
    assert qe.expand("Is there a museum nearby?") == "Is there a museum nearby?"  # 'use' inside 'museum' ignored
    assert QueryExpander.from_file(None).expand("anything") == "anything"
