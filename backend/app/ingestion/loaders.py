"""File discovery and text extraction for supported transcript formats."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class TranscriptLoadError(Exception):
    """Raised when a file cannot be read as a transcript."""


@dataclass(frozen=True)
class DiscoveredFile:
    transcript_id: str
    path: Path
    filename: str
    file_hash: str


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def transcript_id_for(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    return slug or "transcript"


def discover_transcripts(folder: Path) -> list[DiscoveredFile]:
    """Scan a folder (non-recursive) for supported transcript files, sorted naturally by name."""
    if not folder.exists():
        return []

    def natural_key(p: Path):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())]

    files = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith((".", "~$"))),
        key=natural_key,
    )
    seen: set[str] = set()
    out: list[DiscoveredFile] = []
    for p in files:
        tid = transcript_id_for(p)
        if tid in seen:  # e.g. expert_1.txt and expert_1.md
            tid = f"{tid}_{p.suffix.lower().lstrip('.')}"
        seen.add(tid)
        out.append(DiscoveredFile(transcript_id=tid, path=p, filename=p.name, file_hash=file_hash(p)))
    return out


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            data = path.read_bytes()
            for enc in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    return data.decode(enc)
                except UnicodeDecodeError:
                    continue
            raise TranscriptLoadError("Unable to decode text file")
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if suffix == ".docx":
            import docx

            document = docx.Document(str(path))
            return "\n".join(p.text for p in document.paragraphs)
    except TranscriptLoadError:
        raise
    except Exception as exc:  # corrupt pdf/docx etc.
        raise TranscriptLoadError(f"Could not read {path.name}: {exc.__class__.__name__}") from exc
    raise TranscriptLoadError(f"Unsupported file type: {suffix}")
