"""Core RagDag class — Python SDK for ragdag."""

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class SearchResult:
    path: str
    score: float
    content: str = ""
    domain: str = ""
    explain: Optional[dict] = None


@dataclass
class ChunkMeta:
    source: str
    position: int
    total: int
    strategy: str
    hash: str
    heading: str = ""


@dataclass
class AskResult:
    answer: Optional[str]
    context: str
    sources: List[str] = field(default_factory=list)
    confidence: str = "unknown"
    retrieval_attempts: int = 1


@dataclass
class GraphStats:
    domains: int
    documents: int
    chunks: int
    edges: int
    edge_types: dict = field(default_factory=dict)


def _sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return re.sub(r"[^a-z0-9._-]", "", name.lower())


class RagDag:
    """Main ragdag interface for Python."""

    def __init__(self, path: str = "."):
        self._root = Path(path).resolve()
        self._store = self._root / ".ragdag"
        self._ragdag_dir = Path(__file__).parent.parent.parent  # sdk/ragdag -> ragdag root

    @property
    def store_dir(self) -> Path:
        return self._store

    def _config_path(self) -> Path:
        return self._store / ".config"

    def _edges_path(self) -> Path:
        return self._store / ".edges"

    def _processed_path(self) -> Path:
        return self._store / ".processed"

    def _read_config(self, key: str, default: str = "") -> str:
        """Read a config value."""
        config_file = self._config_path()
        if not config_file.exists():
            return default

        section_target, key_target = key.split(".", 1)
        in_section = False
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                in_section = line[1:-1] == section_target
                continue
            if in_section and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == key_target:
                    return v.strip()
        return default

    def _init_store(self):
        """Initialize store structure."""
        self._store.mkdir(parents=True, exist_ok=True)

        config = self._config_path()
        if not config.exists():
            config.write_text(
                "[general]\n"
                "chunk_strategy = heading\n"
                "chunk_size = 1000\n"
                "chunk_overlap = 100\n\n"
                "[embedding]\n"
                "provider = none\n"
                "model = text-embedding-3-small\n"
                "dimensions = 1536\n\n"
                "[llm]\n"
                "provider = none\n"
                "model = gpt-4o-mini\n"
                "max_context = 8000\n\n"
                "[search]\n"
                "default_mode = hybrid\n"
                "top_k = 10\n"
                "keyword_weight = 0.3\n"
                "vector_weight = 0.7\n"
                "rerank = false\n"
                "rerank_model = cross-encoder/ms-marco-MiniLM-L-6-v2\n"
                "hyde = false\n"
                "crag = false\n"
                "crag_max_retries = 1\n\n"
                "[edges]\n"
                "auto_relate = false\n"
                "relate_threshold = 0.8\n"
                "record_queries = false\n\n"
                "[synthesis]\n"
                "enabled = false\n"
                "on_ingest = summary,entities\n"
                "on_query = off\n"
                "on_relate = clusters\n"
                "synthesis_boost = 1.2\n"
            )

        for f in [".edges", ".processed", ".domain-rules"]:
            p = self._store / f
            if not p.exists():
                p.write_text(f"# {f}\n")

    # ------------------------------------------------------------------
    # Ingest (pure Python — no subprocess)
    # ------------------------------------------------------------------

    def add(
        self,
        path: str,
        domain: Optional[str] = None,
        embed: bool = True,
        relate: bool = False,
    ) -> dict:
        """Ingest a file or directory. Pure Python implementation."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        files = []
        if source.is_dir():
            for f in sorted(source.rglob("*")):
                if f.is_file() and not f.name.startswith(".") and ".ragdag" not in f.parts and ".git" not in f.parts:
                    files.append(f)
        else:
            files.append(source)

        chunk_size = int(self._read_config("general.chunk_size", "1000"))
        chunk_overlap = int(self._read_config("general.chunk_overlap", "100"))
        embed_provider = self._read_config("embedding.provider", "none")

        total_files = 0
        total_chunks = 0
        total_skipped = 0

        for file in files:
            abs_path = file.resolve()
            content_hash = _sha256(abs_path)

            # Dedup check
            if self._is_processed(abs_path, content_hash):
                total_skipped += 1
                continue

            # Parse
            try:
                text = self._parse_file(abs_path)
            except Exception:
                text = abs_path.read_text(encoding="utf-8", errors="replace")

            # Select chunk strategy based on file type
            ftype = self._detect_file_type(abs_path)
            config_strategy = self._read_config("general.chunk_strategy", "heading")
            if config_strategy == "proposition":
                strategy = "proposition"
            elif ftype == "markdown":
                strategy = "heading"
            elif ftype == "code":
                strategy = "function"
            elif ftype == "text":
                strategy = "paragraph"
            else:
                strategy = config_strategy

            # Chunk (with metadata)
            chunk_tuples = self._chunk_text_with_meta(text, chunk_size, chunk_overlap, strategy)
            if not chunk_tuples:
                chunk_tuples = [(text, "")]
            chunks = [t[0] for t in chunk_tuples]

            # Determine doc name and domain
            doc_name = _sanitize(file.stem) or "document"
            file_domain = domain or ""
            if file_domain == "auto":
                file_domain = self._apply_domain_rules(str(abs_path)) or "unsorted"

            # Store chunks
            if file_domain:
                target_dir = self._store / file_domain / doc_name
            else:
                target_dir = self._store / doc_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Mark existing synthesis nodes as stale before replacing chunks
            if target_dir.exists():
                self._mark_synthesis_stale(target_dir)

            # Remove old raw chunks (preserve _ synthesis nodes)
            for old in target_dir.glob("*.txt"):
                if not old.name.startswith("_"):
                    old.unlink()

            # Write new chunks with provenance
            total_in_doc = len(chunk_tuples)
            for i, (chunk_text, heading) in enumerate(chunk_tuples, 1):
                chunk_file = target_dir / f"{i:02d}.txt"
                meta = ChunkMeta(
                    source=str(abs_path),
                    heading=heading,
                    position=i,
                    total=total_in_doc,
                    strategy=strategy,
                    hash=content_hash,
                )
                from engines.synthesis import write_chunk_node
                write_chunk_node(chunk_file, chunk_text, meta)

            # Update .processed
            self._record_processed(abs_path, content_hash, file_domain)

            # Create edges
            rel_path = str(target_dir.relative_to(self._store))
            self._create_chunk_edges(rel_path, str(abs_path))

            # Embed
            if embed and embed_provider != "none":
                self._embed_chunks(target_dir, rel_path, file_domain)

            # Synthesis (if enabled)
            if self._read_config("synthesis.enabled", "false") == "true":
                self._synthesize_on_ingest(target_dir, rel_path, file_domain)

            # Update BM25 index incrementally
            try:
                sys.path.insert(0, str(self._ragdag_dir))
                from engines.bm25_index import update_index
                chunk_paths = [
                    str((target_dir / f"{i:02d}.txt").relative_to(self._store))
                    for i in range(1, len(chunks) + 1)
                ]
                update_index(str(self._store), chunk_paths)
            except (ImportError, Exception):
                pass

            total_files += 1
            total_chunks += len(chunks)

        # Build/rebuild edge index after all files processed
        if total_files > 0:
            try:
                sys.path.insert(0, str(self._ragdag_dir))
                from engines.edge_index import build_edge_index
                build_edge_index(str(self._store))
            except (ImportError, Exception):
                pass

        return {"files": total_files, "chunks": total_chunks, "skipped": total_skipped}

    def _detect_file_type(self, path: Path) -> str:
        """Detect file type for strategy selection."""
        ext = path.suffix.lower()
        if ext in (".md", ".markdown"):
            return "markdown"
        if ext in (".txt", ".text", ".log"):
            return "text"
        if ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp",
                    ".h", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
                    ".bash", ".zsh", ".r", ".jl", ".lua", ".pl"):
            return "code"
        if ext == ".csv":
            return "csv"
        if ext in (".json", ".jsonl"):
            return "json"
        if ext == ".pdf":
            return "pdf"
        if ext in (".html", ".htm"):
            return "html"
        if ext == ".docx":
            return "docx"
        return "text"

    def _parse_file(self, path: Path) -> str:
        """Parse a file to plain text."""
        ext = path.suffix.lower()

        if ext in (".md", ".markdown"):
            text = path.read_text(encoding="utf-8", errors="replace")
            # Strip YAML frontmatter
            if text.startswith("---"):
                end = text.find("---", 3)
                if end != -1:
                    text = text[end + 3:].lstrip("\n")
            return text

        if ext == ".pdf":
            return self._parse_pdf(path)

        if ext in (".html", ".htm"):
            return self._parse_html(path)

        if ext == ".docx":
            return self._parse_docx(path)

        if ext == ".csv":
            return self._parse_csv(path)

        if ext in (".json", ".jsonl"):
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                import json
                data = json.loads(text)
                lines = []
                def _flatten(obj, prefix=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            _flatten(v, f"{prefix}{k}.")
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            _flatten(v, f"{prefix}{i}.")
                    else:
                        lines.append(f"{prefix.rstrip('.')}: {obj}")
                _flatten(data)
                return "\n".join(lines)
            except Exception:
                return text

        # Default: read as text (code, config, plain text)
        return path.read_text(encoding="utf-8", errors="replace")

    def _parse_pdf(self, path: Path) -> str:
        """Extract text from PDF using pdftotext."""
        import subprocess
        try:
            result = subprocess.run(
                ["pdftotext", str(path), "-"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: return empty (caller will use raw text or skip)
        raise ValueError(f"pdftotext not available or failed for {path}")

    def _parse_html(self, path: Path) -> str:
        """Extract text from HTML using pandoc or simple tag stripping."""
        import subprocess
        try:
            result = subprocess.run(
                ["pandoc", "-t", "plain", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: strip HTML tags with regex
        text = path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _parse_docx(self, path: Path) -> str:
        """Extract text from DOCX using pandoc."""
        import subprocess
        try:
            result = subprocess.run(
                ["pandoc", "-t", "plain", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        raise ValueError(f"pandoc not available or failed for {path}")

    def _parse_csv(self, path: Path) -> str:
        """Convert CSV to readable key-value text."""
        import csv
        lines = []
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    lines.append(f"--- Record {i} ---")
                    for k, v in row.items():
                        if v:
                            lines.append(f"{k}: {v}")
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(lines)

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Strip YAML frontmatter from chunk or synthesis node text."""
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                return text[end + 5:]
        return text

    def _chunk_text_with_meta(
        self, text: str, chunk_size: int, overlap: int, strategy: str = "heading"
    ) -> List[tuple]:
        """Split text into chunks, returning (text, heading) tuples."""
        if strategy == "heading":
            return self._chunk_heading_with_meta(text, chunk_size, overlap)
        elif strategy == "proposition":
            # Two-pass: heading chunks first, then decompose each
            heading_chunks = self._chunk_heading_with_meta(text, chunk_size, overlap)
            all_props = []
            for chunk_text, heading in heading_chunks:
                props = self._chunk_proposition(chunk_text, chunk_size, overlap)
                # Inherit heading from parent chunk
                all_props.extend([(p_text, heading) for p_text, _ in props])
            return all_props if all_props else heading_chunks
        else:
            plain_chunks = self._chunk_text(text, chunk_size, overlap, strategy)
            return [(c, "") for c in plain_chunks]

    def _chunk_text(
        self, text: str, chunk_size: int, overlap: int, strategy: str = "heading"
    ) -> List[str]:
        """Split text into chunks using the specified strategy."""
        if strategy == "heading":
            return self._chunk_heading(text, chunk_size, overlap)
        elif strategy == "paragraph":
            return self._chunk_paragraph(text, chunk_size, overlap)
        elif strategy == "function":
            return self._chunk_function(text, chunk_size, overlap)
        else:  # fixed
            return self._chunk_fixed(text, chunk_size, overlap)

    def _chunk_heading(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split on markdown headings, with size limit."""
        lines = text.split("\n")
        chunks = []
        buffer = []
        buffer_len = 0

        for line in lines:
            is_header = line.startswith("#")

            if is_header and buffer_len > 0:
                chunk_text = "\n".join(buffer)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                if overlap > 0:
                    buffer = [chunk_text[-overlap:], line]
                else:
                    buffer = [line]
                buffer_len = sum(len(b) for b in buffer)
                continue

            buffer.append(line)
            buffer_len += len(line) + 1

            if buffer_len >= chunk_size:
                chunk_text = "\n".join(buffer)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                if overlap > 0:
                    buffer = [chunk_text[-overlap:]]
                else:
                    buffer = []
                buffer_len = sum(len(b) for b in buffer)

        if buffer:
            chunk_text = "\n".join(buffer)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _chunk_heading_with_meta(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[tuple]:
        """Split on markdown headings, returning (text, heading) tuples."""
        lines = text.split("\n")
        chunks = []
        buffer = []
        buffer_len = 0
        current_heading = ""

        for line in lines:
            is_header = line.startswith("#")

            if is_header:
                if buffer_len > 0:
                    chunk_text = "\n".join(buffer)
                    if chunk_text.strip():
                        chunks.append((chunk_text, current_heading))
                    if overlap > 0:
                        buffer = [chunk_text[-overlap:], line]
                    else:
                        buffer = [line]
                    buffer_len = sum(len(b) for b in buffer)
                current_heading = line.strip()
                if not buffer:
                    buffer = [line]
                    buffer_len = len(line) + 1
                continue

            buffer.append(line)
            buffer_len += len(line) + 1

            if buffer_len >= chunk_size:
                chunk_text = "\n".join(buffer)
                if chunk_text.strip():
                    chunks.append((chunk_text, current_heading))
                if overlap > 0:
                    buffer = [chunk_text[-overlap:]]
                else:
                    buffer = []
                buffer_len = sum(len(b) for b in buffer)

        if buffer:
            chunk_text = "\n".join(buffer)
            if chunk_text.strip():
                chunks.append((chunk_text, current_heading))

        return chunks

    def _chunk_proposition(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[tuple]:
        """Decompose text into atomic propositions via LLM, with sentence-split fallback.

        Returns list of (proposition_text, heading) tuples.
        Heading is empty string -- caller sets from parent chunk if available.
        """
        llm_provider = self._read_config("llm.provider", "none")

        if llm_provider != "none":
            try:
                sys.path.insert(0, str(self._ragdag_dir))
                from engines.llm import call_llm

                llm_model = self._read_config("llm.model", "gpt-4o-mini")
                prompt = (
                    "Decompose this text into self-contained factual statements. "
                    "Each statement should be understandable without context. "
                    "Resolve pronouns and references. Return one statement per line. "
                    "Do not number them or add bullets.\n\n"
                    f"Text:\n{text}"
                )
                system_msg = "You are a text decomposition engine. Return only the propositions, one per line."
                raw = call_llm(system_msg, prompt, llm_provider, llm_model)
                propositions = [
                    line.strip() for line in raw.strip().splitlines()
                    if line.strip() and len(line.strip()) > 10
                ]
                if propositions:
                    return [(p, "") for p in propositions]
            except Exception:
                pass  # Fall through to sentence splitting

        # Fallback: regex sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return [(text, "")]
        return [(s, "") for s in sentences]

    def _chunk_paragraph(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split on blank lines (paragraph boundaries)."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        buffer = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if buffer and len(buffer) + len(para) + 2 > chunk_size:
                chunks.append(buffer)
                if overlap > 0:
                    buffer = buffer[-overlap:] + "\n\n" + para
                else:
                    buffer = para
            elif buffer:
                buffer += "\n\n" + para
            else:
                buffer = para

        if buffer.strip():
            chunks.append(buffer)

        return chunks

    def _chunk_function(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split on function/class boundaries."""
        lines = text.split("\n")
        chunks = []
        buffer = []
        buffer_len = 0

        func_pattern = re.compile(
            r"^(def |class |function |const |let |var |export |pub fn |fn |func )"
        )

        for line in lines:
            is_boundary = bool(func_pattern.match(line.lstrip()))

            if is_boundary and buffer_len > 0:
                chunk_text = "\n".join(buffer)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                if overlap > 0:
                    buffer = [chunk_text[-overlap:], line]
                else:
                    buffer = [line]
                buffer_len = sum(len(b) for b in buffer)
                continue

            buffer.append(line)
            buffer_len += len(line) + 1

            if buffer_len >= chunk_size:
                chunk_text = "\n".join(buffer)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                if overlap > 0:
                    buffer = [chunk_text[-overlap:]]
                else:
                    buffer = []
                buffer_len = sum(len(b) for b in buffer)

        if buffer:
            chunk_text = "\n".join(buffer)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _chunk_fixed(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split by fixed character count."""
        chunks = []
        start = 0
        text_len = len(text)
        # Guard: overlap must be less than chunk_size to guarantee progress
        effective_overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end == text_len:
                break
            start = end - effective_overlap

        return chunks

    def _apply_domain_rules(self, source_path: str) -> str:
        """Apply domain rules to determine domain for a file."""
        rules_file = self._store / ".domain-rules"
        if not rules_file.exists():
            return ""

        source_lower = source_path.lower()
        for line in rules_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: patterns → domain
            if "→" not in line:
                continue
            patterns_str, _, domain_str = line.partition("→")
            domain_str = domain_str.strip()
            if not domain_str:
                continue
            for pattern in patterns_str.split():
                pattern = pattern.strip().lower()
                if pattern and pattern in source_lower:
                    return domain_str
        return ""

    def _is_processed(self, abs_path: Path, content_hash: str) -> bool:
        """Check if file was already processed with same hash."""
        processed = self._processed_path()
        if not processed.exists():
            return False
        marker = f"{abs_path}\t{content_hash}\t"
        return marker in processed.read_text()

    def _record_processed(self, abs_path: Path, content_hash: str, domain: str):
        """Record a processed file."""
        processed = self._processed_path()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Remove old entry
        if processed.exists():
            lines = [
                l for l in processed.read_text().splitlines()
                if not l.startswith(f"{abs_path}\t")
            ]
        else:
            lines = ["# source_path\tcontent_hash\tdomain\ttimestamp"]

        lines.append(f"{abs_path}\t{content_hash}\t{domain}\t{timestamp}")
        processed.write_text("\n".join(lines) + "\n")

    def _create_chunk_edges(self, doc_rel_path: str, source_path: str):
        """Create chunked_from edges."""
        edges_file = self._edges_path()
        target_dir = self._store / doc_rel_path

        # Remove old edges for this source
        if edges_file.exists():
            lines = [
                l for l in edges_file.read_text().splitlines()
                if not (f"\t{source_path}\tchunked_from" in l)
            ]
        else:
            lines = ["# source\ttarget\tedge_type\tmetadata"]

        # Add new edges
        for chunk in sorted(target_dir.glob("*.txt")):
            chunk_rel = str(chunk.relative_to(self._store))
            lines.append(f"{chunk_rel}\t{source_path}\tchunked_from\t")

        edges_file.write_text("\n".join(lines) + "\n")

    def _embed_chunks(self, chunk_dir: Path, doc_prefix: str, domain: str):
        """Embed chunks using configured engine."""
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.embeddings import write_embeddings

            provider = self._read_config("embedding.provider", "none")
            model = self._read_config("embedding.model", "text-embedding-3-small")
            dims = int(self._read_config("embedding.dimensions", "1536"))

            if provider == "openai":
                from engines.openai_engine import OpenAIEngine
                engine = OpenAIEngine(model=model, dims=dims)
            elif provider == "local":
                from engines.local_engine import LocalEngine
                engine = LocalEngine(model=model, dims=dims)
            else:
                return

            texts, paths = [], []
            for f in sorted(chunk_dir.glob("*.txt")):
                t = f.read_text(encoding="utf-8").strip()
                if t:
                    texts.append(t)
                    paths.append(f"{doc_prefix}/{f.name}")

            if not texts:
                return

            vectors = engine.embed(texts)

            embed_dir = self._store / domain if domain else self._store
            embed_dir.mkdir(parents=True, exist_ok=True)

            write_embeddings(
                output_dir=str(embed_dir),
                vectors=vectors,
                chunk_paths=paths,
                dimensions=engine.dimensions(),
                model_name_str=engine.model_name(),
                append=True,
            )
        except Exception:
            pass  # Embedding failure doesn't block ingest

    def _synthesize_on_ingest(self, target_dir: Path, doc_rel_path: str, domain: str):
        """Run ingest-time synthesis: summary + entity extraction."""
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.synthesis import (
                summarize_chunks,
                extract_entities,
                write_synthesis_node,
                chunk_file_hash,
                read_frontmatter,
                read_body,
            )

            provider = self._read_config("llm.provider", "none")
            model = self._read_config("llm.model", "gpt-4o-mini")
            on_ingest = self._read_config("synthesis.on_ingest", "summary,entities")

            if provider == "none":
                return

            # Read raw chunks (non _ prefixed)
            chunk_files = sorted(
                f for f in target_dir.glob("*.txt") if not f.name.startswith("_")
            )
            if not chunk_files:
                return

            chunks = []
            source_paths = []
            source_hashes = []
            for f in chunk_files:
                chunks.append(f.read_text(encoding="utf-8"))
                source_paths.append(str(f.relative_to(self._store)))
                source_hashes.append(chunk_file_hash(f))

            edges_file = self._edges_path()
            new_edges = []

            # Summary
            if "summary" in on_ingest:
                summary = summarize_chunks(chunks, provider, model)
                summary_path = target_dir / "_summary.txt"
                write_synthesis_node(
                    summary_path, summary, "summary", source_paths, source_hashes
                )
                summary_rel = str(summary_path.relative_to(self._store))
                for src in source_paths:
                    new_edges.append(f"{summary_rel}\t{src}\tderived_from\tingest")

            # Entity extraction
            if "entities" in on_ingest:
                entities = extract_entities(chunks, provider, model)
                for ent in entities:
                    name = _sanitize(ent.get("name", "unknown").replace(" ", "_"))[:50]
                    etype = ent.get("type", "entity")
                    desc = ent.get("description", "")
                    node_name = f"_{etype}_{name}.txt"
                    node_path = target_dir / node_name

                    if node_path.exists():
                        existing_fm = read_frontmatter(node_path)
                        existing_body = read_body(node_path)
                        if existing_fm:
                            merged_sources = list(set(
                                existing_fm.get("sources", []) + source_paths
                            ))
                            merged_hashes = list(set(
                                existing_fm.get("source_hashes", []) + source_hashes
                            ))
                            merged_body = existing_body.strip() + "\n\n" + desc
                            write_synthesis_node(
                                node_path, merged_body, etype,
                                merged_sources, merged_hashes,
                            )
                        else:
                            write_synthesis_node(
                                node_path, desc, etype, source_paths, source_hashes
                            )
                    else:
                        write_synthesis_node(
                            node_path, desc, etype, source_paths, source_hashes
                        )

                    node_rel = str(node_path.relative_to(self._store))
                    for src in source_paths:
                        new_edges.append(f"{node_rel}\t{src}\tderived_from\tingest")

            # Append edges
            if new_edges:
                with open(edges_file, "a") as f:
                    for edge in new_edges:
                        f.write(edge + "\n")

            # Embed synthesis nodes
            self._embed_synthesis_nodes(target_dir, doc_rel_path, domain)

        except Exception:
            pass  # Synthesis failure doesn't block ingest

    def _mark_synthesis_stale(self, target_dir: Path):
        """Mark all synthesis nodes in a directory as stale."""
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.synthesis import mark_stale, is_synthesis_node
            for f in target_dir.glob("_*.txt"):
                if is_synthesis_node(f):
                    mark_stale(f)
        except Exception:
            pass

        # Also check _queries and _synthesis for nodes referencing this dir
        try:
            from engines.synthesis import read_frontmatter, mark_stale as _mark

            doc_prefix = str(target_dir.relative_to(self._store))
            for synth_dir_name in ("_queries", "_synthesis"):
                synth_dir = self._store / synth_dir_name
                if not synth_dir.exists():
                    continue
                for f in synth_dir.glob("_*.txt"):
                    fm = read_frontmatter(f)
                    if fm and any(s.startswith(doc_prefix + "/") for s in fm.get("sources", [])):
                        _mark(f)
        except Exception:
            pass

    def _embed_synthesis_nodes(self, doc_dir: Path, doc_prefix: str, domain: str):
        """Embed _ prefixed synthesis nodes."""
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.embeddings import write_embeddings
            from engines.synthesis import read_body

            provider = self._read_config("embedding.provider", "none")
            model = self._read_config("embedding.model", "text-embedding-3-small")
            dims = int(self._read_config("embedding.dimensions", "1536"))

            if provider == "openai":
                from engines.openai_engine import OpenAIEngine
                engine = OpenAIEngine(model=model, dims=dims)
            elif provider == "local":
                from engines.local_engine import LocalEngine
                engine = LocalEngine(model=model, dims=dims)
            else:
                return

            texts, paths = [], []
            for f in sorted(doc_dir.glob("_*.txt")):
                t = read_body(f).strip()
                if t:
                    texts.append(t)
                    paths.append(f"{doc_prefix}/{f.name}")

            if not texts:
                return

            vectors = engine.embed(texts)

            embed_dir = self._store / domain if domain else self._store
            embed_dir.mkdir(parents=True, exist_ok=True)

            write_embeddings(
                output_dir=str(embed_dir),
                vectors=vectors,
                chunk_paths=paths,
                dimensions=engine.dimensions(),
                model_name_str=engine.model_name(),
                append=True,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Search (pure Python)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        domain: Optional[str] = None,
        top: int = 10,
        explain: bool = False,
    ) -> List[SearchResult]:
        """Search the corpus.

        Args:
            query: Search query string.
            mode: "keyword" (BM25), "vector" (embeddings), or "hybrid" (BM25 + vector
                  fused with Reciprocal Rank Fusion). Default: "hybrid".
            domain: Optional domain filter.
            top: Number of results to return.
            explain: If True, each SearchResult.explain contains a dict with score
                     breakdown: {"bm25": float, "vector": float, "rrf": float,
                     "reranker": float (if enabled)}.

        Returns:
            List of SearchResult sorted by relevance score descending.
        """
        if mode == "keyword":
            return self._keyword_search(query, domain, top, explain=explain)

        try:
            return self._python_search(query, mode, domain, top, explain=explain)
        except Exception:
            return self._keyword_search(query, domain, top, explain=explain)

    def _synthesis_boost(self) -> float:
        """Get the synthesis boost factor from config."""
        return float(self._read_config("synthesis.synthesis_boost", "1.2"))

    def _keyword_search(
        self, query: str, domain: Optional[str], top: int,
        explain: bool = False,
    ) -> List[SearchResult]:
        """BM25 keyword search over flat files."""
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.bm25 import bm25_search

            bm25_results = bm25_search(
                str(self._store), query, domain or "", top,
                synthesis_boost=self._synthesis_boost(),
            )
        except ImportError:
            return self._keyword_search_legacy(query, domain, top)

        results = []
        for path, score in bm25_results:
            full_path = self._store / path
            content = ""
            if full_path.exists():
                raw = full_path.read_text(encoding="utf-8")
                content = self._strip_frontmatter(raw)
            parts = path.split("/")
            domain_name = parts[0] if len(parts) >= 3 else ""
            explain_data = {"bm25": round(score, 6)} if explain else None
            results.append(SearchResult(
                path=path, score=score, content=content,
                domain=domain_name, explain=explain_data,
            ))
        return results

    def _keyword_search_legacy(
        self, query: str, domain: Optional[str], top: int
    ) -> List[SearchResult]:
        """Legacy pure Python keyword search (fallback if engines not available)."""
        search_path = self._store / domain if domain else self._store
        query_lower = query.lower()
        words = [w for w in query_lower.split() if len(w) >= 2]
        boost = self._synthesis_boost()

        results = []
        for txt_file in search_path.rglob("*.txt"):
            if txt_file.name.startswith("."):
                continue
            try:
                content = txt_file.read_text(encoding="utf-8")
            except Exception:
                continue

            is_synth = txt_file.name.startswith("_")
            search_content = content
            if is_synth and content.startswith("---\n"):
                end = content.find("\n---\n", 4)
                if end != -1:
                    search_content = content[end + 5:]

            content_lower = search_content.lower()
            content_len = len(content_lower)
            if content_len == 0:
                continue

            match_count = sum(content_lower.count(word) for word in words)

            if match_count > 0:
                score = match_count / content_len
                if is_synth:
                    if "stale: true" in content:
                        score *= boost * 0.5
                    else:
                        score *= boost
                rel_path = str(txt_file.relative_to(self._store))
                parts = rel_path.split("/")
                domain_name = parts[0] if len(parts) >= 3 else ""
                results.append(SearchResult(
                    path=rel_path, score=score, content=search_content,
                    domain=domain_name,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top]

    def _hyde_expand(self, query: str) -> str:
        """Expand query into a hypothetical answer for better vector matching.

        Returns the original query if HyDE is disabled, LLM is unavailable,
        or the LLM call fails.
        """
        hyde_enabled = self._read_config("search.hyde", "false") == "true"
        if not hyde_enabled:
            return query

        llm_provider = self._read_config("llm.provider", "none")
        if llm_provider == "none":
            return query

        # Check cache
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_dir = self._store / ".hyde_cache"
        cache_file = cache_dir / f"{query_hash}.txt"

        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        # Generate hypothetical answer
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.llm import call_llm

            llm_model = self._read_config("llm.model", "gpt-4o-mini")
            system_msg = (
                "You are a document retrieval assistant. Write a short factual "
                "passage that would answer the given question. Do not add disclaimers."
            )
            prompt = (
                f"Write a short passage (3-4 sentences) that would answer "
                f"this question:\n{query}"
            )
            hypothetical = call_llm(system_msg, prompt, llm_provider, llm_model)

            if hypothetical and hypothetical.strip():
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(hypothetical.strip(), encoding="utf-8")
                return hypothetical.strip()
        except Exception:
            pass

        return query

    def _python_search(
        self, query: str, mode: str, domain: Optional[str], top: int,
        explain: bool = False,
    ) -> List[SearchResult]:
        """Vector/hybrid search using Python engines."""
        sys.path.insert(0, str(self._ragdag_dir))
        from engines.similarity import search_vectors

        provider = self._read_config("embedding.provider", "none")
        if provider == "none":
            return self._keyword_search(query, domain, top, explain=explain)

        model = self._read_config("embedding.model", "text-embedding-3-small")
        dims = int(self._read_config("embedding.dimensions", "1536"))

        if provider == "openai":
            from engines.openai_engine import OpenAIEngine
            engine = OpenAIEngine(model=model, dims=dims)
        elif provider == "local":
            from engines.local_engine import LocalEngine
            engine = LocalEngine(model=model, dims=dims)
        else:
            return self._keyword_search(query, domain, top, explain=explain)

        # HyDE: embed hypothetical answer for vector search, keep original for BM25
        hyde_query = self._hyde_expand(query)
        query_vec = engine.embed([hyde_query])[0]

        if mode == "hybrid":
            from engines.bm25 import bm25_search
            from engines.rrf import reciprocal_rank_fusion

            # Phase 1: BM25 keyword search
            bm25_results = bm25_search(
                str(self._store), query, domain or "", top * 3,
                synthesis_boost=self._synthesis_boost(),
            )

            # Phase 2: Vector search
            candidate_paths = [p for p, _ in bm25_results] if bm25_results else None
            vec_results = search_vectors(
                query_embedding=query_vec,
                store_dir=str(self._store),
                domain=domain or "",
                top_k=top * 3,
                candidate_paths=candidate_paths,
            )

            # Phase 3: RRF fusion
            fused = reciprocal_rank_fusion(
                [bm25_results, vec_results], k=60, top_k=top
            )

            # Build explain data
            explain_map = {}
            if explain:
                bm25_map = {p: s for p, s in bm25_results}
                vec_map = {p: s for p, s in vec_results}
                for path, rrf_score in fused:
                    explain_map[path] = {
                        "bm25": round(bm25_map.get(path, 0.0), 6),
                        "vector": round(vec_map.get(path, 0.0), 6),
                        "rrf": round(rrf_score, 6),
                    }

            # Phase 4: Optional reranking
            rerank_enabled = self._read_config("search.rerank", "false") == "true"
            if rerank_enabled and fused:
                try:
                    from engines.reranker import rerank as do_rerank
                    candidates = []
                    for path, score in fused:
                        fp = self._store / path
                        content = fp.read_text(encoding="utf-8") if fp.exists() else ""
                        candidates.append((path, score, content))
                    reranked = do_rerank(query, candidates, top_k=top)
                    if explain:
                        for path, blended in reranked:
                            if path in explain_map:
                                explain_map[path]["reranker"] = round(blended, 6)
                    fused = reranked
                except Exception:
                    pass

            final_results = fused
        else:
            # Pure vector search
            final_results = search_vectors(
                query_embedding=query_vec,
                store_dir=str(self._store),
                domain=domain or "",
                top_k=top,
            )
            explain_map = {}
            if explain:
                for path, score in final_results:
                    explain_map[path] = {"vector": round(score, 6)}

        return [
            SearchResult(
                path=path,
                score=score,
                content=self._strip_frontmatter((self._store / path).read_text(encoding="utf-8")) if (self._store / path).exists() else "",
                domain=path.split("/")[0] if len(path.split("/")) >= 3 else "",
                explain=explain_map.get(path) if explain else None,
            )
            for path, score in final_results
        ]

    # ------------------------------------------------------------------
    # CRAG helpers
    # ------------------------------------------------------------------

    def _crag_relevance_check(self, question: str, context: str) -> tuple:
        """Check if retrieved context is relevant to the question.

        Returns (rating, gap_description).
        """
        sys.path.insert(0, str(self._ragdag_dir))
        from engines.llm import call_llm

        llm_provider = self._read_config("llm.provider", "none")
        llm_model = self._read_config("llm.model", "gpt-4o-mini")

        context_preview = context[:2000]

        system_msg = (
            "You are a relevance evaluator. Rate whether the provided context "
            "can answer the question. Respond with exactly one of:\n"
            "SUFFICIENT\n"
            "PARTIAL: <what information is missing>\n"
            "INSUFFICIENT\n"
            "Nothing else."
        )
        user_msg = f"Question: {question}\n\nContext:\n{context_preview}"

        try:
            raw = call_llm(system_msg, user_msg, llm_provider, llm_model)
            raw = raw.strip() if raw else "INSUFFICIENT"
            upper = raw.upper()

            if upper.startswith("SUFFICIENT"):
                return ("sufficient", "")
            elif upper.startswith("PARTIAL"):
                gap = raw.split(":", 1)[1].strip() if ":" in raw else ""
                return ("partial", gap)
            else:
                return ("insufficient", "")
        except Exception:
            return ("sufficient", "")

    def _crag_reformulate(self, question: str, gap: str) -> str:
        """Generate a reformulated search query based on identified gaps."""
        sys.path.insert(0, str(self._ragdag_dir))
        from engines.llm import call_llm

        llm_provider = self._read_config("llm.provider", "none")
        llm_model = self._read_config("llm.model", "gpt-4o-mini")

        system_msg = "Generate a concise search query to find the missing information. Return only the query, nothing else."
        if gap:
            user_msg = f"Original question: {question}\nMissing information: {gap}\nSearch query:"
        else:
            user_msg = f"Rephrase this question as a search query with key terms:\n{question}"

        try:
            result = call_llm(system_msg, user_msg, llm_provider, llm_model)
            return result.strip() if result else question
        except Exception:
            return question

    def _expand_via_graph(self, results: list, seen_paths: set) -> list:
        """Pull in related chunks via graph edges."""
        edges_file = self._edges_path()
        expanded = []
        expansion_types = ("related_to", "references", "derived_from")

        use_index = False
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.edge_index import load_edge_index, lookup_edges
            if self._edge_index_is_fresh() and load_edge_index(str(self._store)) is not None:
                use_index = True
        except ImportError:
            pass

        if use_index:
            for r in results:
                node_edges = lookup_edges(str(self._store), r.path)
                if not node_edges:
                    continue
                for e in node_edges:
                    if e["direction"] == "outgoing" and e["edge_type"] in expansion_types:
                        target = e["node"]
                        if target not in seen_paths:
                            target_file = self._store / target
                            if target_file.exists():
                                content = target_file.read_text(encoding="utf-8")
                                content = self._strip_frontmatter(content)
                                expanded.append(SearchResult(
                                    path=target, score=r.score * 0.8,
                                    content=content,
                                    domain=target.split("/")[0] if len(target.split("/")) >= 3 else "",
                                ))
                                seen_paths.add(target)
        elif edges_file.exists():
            edges_text = edges_file.read_text()
            for r in results:
                for line in edges_text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    source, target, etype = parts[0], parts[1], parts[2]
                    if source == r.path and etype in expansion_types:
                        if target not in seen_paths:
                            target_file = self._store / target
                            if target_file.exists():
                                content = target_file.read_text(encoding="utf-8")
                                content = self._strip_frontmatter(content)
                                expanded.append(SearchResult(
                                    path=target, score=r.score * 0.8,
                                    content=content,
                                    domain=target.split("/")[0] if len(target.split("/")) >= 3 else "",
                                ))
                                seen_paths.add(target)
        return expanded

    def _build_context(self, results: list) -> tuple:
        """Assemble context string from search results, respecting token limits."""
        max_context = int(self._read_config("llm.max_context", "8000"))
        context_parts = []
        sources = []
        tokens_used = 0

        for r in results:
            chunk_tokens = int(len(r.content.split()) * 1.3)
            if tokens_used + chunk_tokens > max_context:
                break
            context_parts.append(
                f"--- Source: {r.path} (score: {r.score:.4f}) ---\n{r.content}"
            )
            sources.append(r.path)
            tokens_used += chunk_tokens

        return "\n\n".join(context_parts), sources

    # ------------------------------------------------------------------
    # Ask (RAG)
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        domain: Optional[str] = None,
        use_llm: bool = True,
    ) -> AskResult:
        """Ask a question using RAG, with optional CRAG validation loop."""
        crag_enabled = (
            self._read_config("search.crag", "false") == "true"
            and use_llm
            and self._read_config("llm.provider", "none") != "none"
        )
        max_retries = int(self._read_config("search.crag_max_retries", "1"))
        current_query = question
        all_results: list = []
        seen_paths: set = set()
        attempts = 0
        confidence = "unknown"
        context = ""
        sources: List[str] = []

        for attempt in range(1 + max_retries):
            attempts = attempt + 1

            # Search
            results = self.search(current_query, mode="hybrid", domain=domain, top=10)

            # Merge with previous results (dedup by path)
            for r in results:
                if r.path not in seen_paths:
                    all_results.append(r)
                    seen_paths.add(r.path)

            if not all_results:
                if crag_enabled and attempt < max_retries:
                    current_query = self._crag_reformulate(question, "no results found")
                    continue
                return AskResult(
                    answer=None, context="", sources=[],
                    confidence="insufficient" if crag_enabled else "unknown",
                    retrieval_attempts=attempts,
                )

            # Graph expansion
            expanded = self._expand_via_graph(all_results[:5], set(seen_paths))
            working_results = all_results + expanded
            for e in expanded:
                seen_paths.add(e.path)

            # Build context
            context, sources = self._build_context(working_results)

            # CRAG relevance check
            if crag_enabled:
                rating, gap = self._crag_relevance_check(question, context)
                confidence = rating

                if rating == "sufficient" or attempt >= max_retries:
                    break
                elif rating in ("partial", "insufficient"):
                    current_query = self._crag_reformulate(question, gap)
                    continue
            else:
                break

        if not use_llm:
            return AskResult(
                answer=None, context=context, sources=sources,
                confidence=confidence, retrieval_attempts=attempts,
            )

        llm_provider = self._read_config("llm.provider", "none")
        if llm_provider == "none":
            return AskResult(
                answer=None, context=context, sources=sources,
                confidence=confidence, retrieval_attempts=attempts,
            )

        sys.path.insert(0, str(self._ragdag_dir))
        from engines.llm import get_answer

        llm_model = self._read_config("llm.model", "gpt-4o-mini")

        answer = get_answer(
            question=question,
            context=context,
            provider=llm_provider,
            model=llm_model,
        )

        # File answer back into the store (if enabled)
        on_query = self._read_config("synthesis.on_query", "off")
        if on_query != "off" and answer:
            self._file_answer(question, answer, sources)

        return AskResult(
            answer=answer, context=context, sources=sources,
            confidence=confidence, retrieval_attempts=attempts,
        )

    def _file_answer(self, question: str, answer: str, sources: List[str]):
        """File an answer back into the store as a synthesis node."""
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.synthesis import write_synthesis_node, content_hash, chunk_file_hash

            queries_dir = self._store / "_queries"
            queries_dir.mkdir(parents=True, exist_ok=True)

            answer_hash = content_hash(question)
            answer_path = queries_dir / f"_answer_{answer_hash}.txt"

            source_hashes = []
            for src in sources:
                src_path = self._store / src
                if src_path.exists():
                    source_hashes.append(chunk_file_hash(src_path))
                else:
                    source_hashes.append("")

            write_synthesis_node(
                answer_path,
                f"Question: {question}\n\nAnswer: {answer}",
                "answer",
                sources,
                source_hashes,
            )

            # Add edges
            edges_file = self._edges_path()
            answer_rel = str(answer_path.relative_to(self._store))
            with open(edges_file, "a") as f:
                for src in sources:
                    f.write(f"{answer_rel}\t{src}\tderived_from\tquery\n")

            # Embed the answer node
            self._embed_synthesis_nodes(queries_dir, "_queries", "")

        except Exception:
            pass  # Answer filing failure doesn't block the response

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def graph(self, domain: Optional[str] = None) -> GraphStats:
        """Get graph summary statistics."""
        domains = 0
        documents = 0
        chunks = 0

        for d in self._store.iterdir():
            if not d.is_dir() or d.name.startswith("."):
                continue
            if domain and d.name != domain:
                continue
            domains += 1
            for doc_dir in d.iterdir():
                if not doc_dir.is_dir():
                    continue
                documents += 1
                chunks += len(list(doc_dir.glob("*.txt")))

        edges_file = self._edges_path()
        edge_types = {}
        total_edges = 0
        if edges_file.exists():
            for line in edges_file.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    total_edges += 1
                    etype = parts[2]
                    edge_types[etype] = edge_types.get(etype, 0) + 1

        return GraphStats(
            domains=domains, documents=documents, chunks=chunks,
            edges=total_edges, edge_types=edge_types,
        )

    def neighbors(self, node_path: str) -> List[dict]:
        """Get connected nodes."""
        # Fast path: use edge index (only if index is up to date)
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.edge_index import lookup_edges
            idx_path = self._store / "_edge_index.json"
            edges_file = self._edges_path()
            # Only use index if it's at least as new as .edges
            if idx_path.exists() and edges_file.exists() and idx_path.stat().st_mtime >= edges_file.stat().st_mtime:
                result = lookup_edges(str(self._store), node_path)
                if result is not None:
                    return result
        except (ImportError, OSError):
            pass

        # Slow path: scan .edges file
        edges_file = self._edges_path()
        if not edges_file.exists():
            return []

        results = []
        for line in edges_file.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            source, target, etype = parts[0], parts[1], parts[2]
            metadata = parts[3] if len(parts) > 3 else ""

            if source == node_path:
                results.append({
                    "direction": "outgoing", "node": target,
                    "edge_type": etype, "metadata": metadata,
                })
            elif target == node_path:
                results.append({
                    "direction": "incoming", "node": source,
                    "edge_type": etype, "metadata": metadata,
                })
        return results

    def _edge_index_is_fresh(self) -> bool:
        """Check if edge index exists and is at least as new as .edges."""
        try:
            idx_path = self._store / "_edge_index.json"
            edges_file = self._edges_path()
            return (idx_path.exists() and edges_file.exists()
                    and idx_path.stat().st_mtime >= edges_file.stat().st_mtime)
        except OSError:
            return False

    def trace(self, node_path: str) -> List[dict]:
        """Get provenance chain."""
        # Try fast path via edge index (only if fresh)
        edge_index = None
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.edge_index import load_edge_index
            if self._edge_index_is_fresh() and load_edge_index(str(self._store)) is not None:
                edge_index = True
        except ImportError:
            pass

        edges_file = self._edges_path()
        if not edge_index and not edges_file.exists():
            return []

        chain = []
        current = node_path
        visited = set()
        provenance_types = ("chunked_from", "derived_via", "derived_from", "synthesizes")

        while current not in visited:
            visited.add(current)
            parent = None

            if edge_index:
                from engines.edge_index import lookup_edges
                edges = lookup_edges(str(self._store), current)
                if edges:
                    for e in edges:
                        if e["direction"] == "outgoing" and e["edge_type"] in provenance_types:
                            parent = e["node"]
                            chain.append({"node": current, "parent": parent, "edge_type": e["edge_type"]})
                            break
            else:
                for line in edges_file.read_text().splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    source, target, etype = parts[0], parts[1], parts[2]
                    if source == current and etype in provenance_types:
                        parent = target
                        chain.append({"node": current, "parent": target, "edge_type": etype})
                        break

            if parent is None:
                chain.append({"node": current, "parent": None, "edge_type": "origin"})
                break
            current = parent

        return chain

    def relate(self, domain: Optional[str] = None, threshold: float = 0.8) -> str:
        """Compute semantic edges between chunks."""
        sys.path.insert(0, str(self._ragdag_dir))
        from engines import relate_cli

        argv = ["relate_cli", "--store-dir", str(self._store), "--threshold", str(threshold)]
        if domain:
            argv += ["--domain", domain]

        old_argv = sys.argv
        sys.argv = argv
        try:
            relate_cli.main()
        finally:
            sys.argv = old_argv
        return "Done"

    def reindex(self, what: str = "all") -> None:
        """Rebuild indexes from source data.

        Args:
            what: 'bm25', 'edges', or 'all'.
        """
        sys.path.insert(0, str(self._ragdag_dir))
        if what in ("bm25", "all"):
            from engines.bm25_index import build_index
            build_index(str(self._store))
        if what in ("edges", "all"):
            from engines.edge_index import build_edge_index
            build_edge_index(str(self._store))

    def link(self, source: str, target: str, edge_type: str = "references"):
        """Create a manual edge."""
        edges_file = self._edges_path()
        with open(edges_file, "a") as f:
            f.write(f"{source}\t{target}\t{edge_type}\t\n")

        # Update edge index if it exists
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.edge_index import append_edge
            append_edge(str(self._store), source, target, edge_type, "")
        except (ImportError, Exception):
            pass
