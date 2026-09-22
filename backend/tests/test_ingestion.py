import shutil

from conftest import FIXTURES, make_settings

from app.services.state import AppState


def test_initial_scan_indexes_all_files(app_state):
    ids = {t.id for t in app_state.store.list_transcripts()}
    assert ids == {"alpha", "beta"}
    stats = app_state.stats()
    assert stats.transcripts == 2 and stats.experts == 2 and stats.markets == 2
    assert stats.evidence_segments == 9


def test_unchanged_files_are_skipped(app_state):
    report = app_state.refresh()
    assert sorted(report.unchanged) == ["alpha", "beta"]
    assert not report.added and not report.updated and not report.removed


def test_new_file_is_discovered_without_code_changes(app_state, transcripts_dir):
    raw = (FIXTURES / "alpha.txt").read_text(encoding="utf-8")
    new = raw.replace("Dr. Clara Alvarez", "Dr. Ines Costa").replace("Dr. Alvarez", "Dr. Costa").replace("Northland", "Westmark")
    (transcripts_dir / "expert_4.txt").write_text(new, encoding="utf-8")
    report = app_state.refresh()
    assert report.added == ["expert_4"]
    experts = {e.expert_name for e in app_state.store.experts()}
    assert "Dr. Ines Costa" in experts and app_state.stats().markets == 3


def test_modified_file_is_reprocessed(app_state, transcripts_dir):
    path = transcripts_dir / "alpha.txt"
    path.write_text(path.read_text(encoding="utf-8").replace("6 to 12 months", "8 to 14 months"), encoding="utf-8")
    report = app_state.refresh()
    assert report.updated == ["alpha"] and report.unchanged == ["beta"]
    assert "8 to 14 months" in app_state.store.get_one("alpha_04_09").text


def test_deleted_file_is_removed_from_index(app_state, transcripts_dir):
    (transcripts_dir / "beta.md").unlink()
    report = app_state.refresh()
    assert report.removed == ["beta"]
    assert app_state.store.get_one("beta_00_18") is None
    assert all(h.transcript_id == "alpha" for h in app_state.retriever.retrieve("training capacity"))


def test_invalid_and_empty_files_fail_gracefully(app_state, transcripts_dir):
    shutil.copy(FIXTURES / "blank.txt", transcripts_dir / "blank.txt")
    (transcripts_dir / "broken.pdf").write_bytes(b"not a pdf")
    report = app_state.refresh()
    failed = {f["transcript_id"] for f in report.failed}
    assert failed == {"blank", "broken"}
    assert {e.transcript_id for e in app_state.store.experts()} == {"alpha", "beta"}


def test_empty_folder(tmp_path):
    state = AppState(make_settings(tmp_path))
    report = state.refresh()
    assert report.total == 0
    assert state.stats().transcripts == 0
    from app.models.schemas import NO_EVIDENCE_MESSAGE, AskRequest

    assert state.research.ask(AskRequest(question="What are the barriers?")).answer == NO_EVIDENCE_MESSAGE
    assert state.research.insights().by_question == []
    state.db.close()


def test_force_reindex_and_metadata_override_survives(app_state):
    assert app_state.indexer.update_metadata("alpha", {"role": "Chief of Surgery"})
    app_state.retriever.invalidate()
    report = app_state.refresh()
    assert "alpha" in report.unchanged
    report = app_state.indexer.sync(app_state.settings.transcripts_dir, force=True)
    assert sorted(report.updated) == ["alpha", "beta"]
    assert app_state.store.get_one("alpha_02_08").role == "Chief of Surgery"


def test_unsupported_extensions_ignored(app_state, transcripts_dir):
    (transcripts_dir / "notes.csv").write_text("a,b", encoding="utf-8")
    assert app_state.refresh().total == 2
