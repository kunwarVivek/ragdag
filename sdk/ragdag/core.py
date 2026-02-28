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


@dataclass
class AskResult:
    answer: Optional[str]
    context: str
    sources: List[str] = field(default_factory=list)


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
                "vector_weight = 0.7\n\n"
                "[edges]\n"
                "auto_relate = false\n"
                "relate_threshold = 0.8\n"
                "record_queries = false\n"
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
            if ftype == "markdown":
                strategy = "heading"
            elif ftype == "code":
                strategy = "function"
            elif ftype == "text":
                strategy = "paragraph"
            else:
                strategy = config_strategy

            # Chunk
            chunks = self._chunk_text(text, chunk_size, chunk_overlap, strategy)
            if not chunks:
                chunks = [text]

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

            # Remove old chunks
            for old in target_dir.glob("*.txt"):
                old.unlink()

            # Write new chunks
            for i, chunk_text in enumerate(chunks, 1):
                chunk_file = target_dir / f"{i:02d}.txt"
                chunk_file.write_text(chunk_text, encoding="utf-8")

            # Update .processed
            self._record_processed(abs_path, content_hash, file_domain)

            # Create edges
            rel_path = str(target_dir.relative_to(self._store))
            self._create_chunk_edges(rel_path, str(abs_path))

            # Embed
            if embed and embed_provider != "none":
                self._embed_chunks(target_dir, rel_path, file_domain)

            total_files += 1
            total_chunks += len(chunks)

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

    # ------------------------------------------------------------------
    # Search (pure Python)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        domain: Optional[str] = None,
        top: int = 10,
    ) -> List[SearchResult]:
        """Search the corpus."""
        if mode == "keyword":
            return self._keyword_search(query, domain, top)

        try:
            return self._python_search(query, mode, domain, top)
        except Exception:
            return self._keyword_search(query, domain, top)

    def _keyword_search(
        self, query: str, domain: Optional[str], top: int
    ) -> List[SearchResult]:
        """Pure Python keyword search."""
        search_path = self._store / domain if domain else self._store
        query_lower = query.lower()
        words = [w for w in query_lower.split() if len(w) >= 2]

        results = []
        for txt_file in search_path.rglob("*.txt"):
            if txt_file.name.startswith("_"):
                continue
            try:
                content = txt_file.read_text(encoding="utf-8")
            except Exception:
                continue

            content_lower = content.lower()
            content_len = len(content_lower)
            if content_len == 0:
                continue

            match_count = sum(content_lower.count(word) for word in words)

            if match_count > 0:
                score = match_count / content_len
                rel_path = str(txt_file.relative_to(self._store))
                parts = rel_path.split("/")
                domain_name = parts[0] if len(parts) >= 3 else ""
                results.append(SearchResult(
                    path=rel_path, score=score, content=content, domain=domain_name,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top]

    def _python_search(
        self, query: str, mode: str, domain: Optional[str], top: int
    ) -> List[SearchResult]:
        """Vector/hybrid search using Python engines."""
        sys.path.insert(0, str(self._ragdag_dir))
        from engines.similarity import search_vectors

        provider = self._read_config("embedding.provider", "none")
        if provider == "none":
            return self._keyword_search(query, domain, top)

        model = self._read_config("embedding.model", "text-embedding-3-small")
        dims = int(self._read_config("embedding.dimensions", "1536"))

        if provider == "openai":
            from engines.openai_engine import OpenAIEngine
            engine = OpenAIEngine(model=model, dims=dims)
        elif provider == "local":
            from engines.local_engine import LocalEngine
            engine = LocalEngine(model=model, dims=dims)
        else:
            return self._keyword_search(query, domain, top)

        query_vec = engine.embed([query])[0]

        if mode == "hybrid":
            # Hybrid: keyword pre-filter + vector + score fusion
            kw_results = self._keyword_search(query, domain, top * 3)
            kw_scores = {r.path: r.score for r in kw_results}
            candidates = [r.path for r in kw_results]

            vec_results = search_vectors(
                query_embedding=query_vec,
                store_dir=str(self._store),
                domain=domain or "",
                top_k=top * 2,
                candidate_paths=candidates,
            )

            # Score fusion with configurable weights
            kw_weight = float(self._read_config("search.keyword_weight", "0.3"))
            vec_weight = float(self._read_config("search.vector_weight", "0.7"))

            # Normalize scores to [0,1]
            max_kw = max(kw_scores.values()) if kw_scores else 1.0
            max_vec = max(s for _, s in vec_results) if vec_results else 1.0

            fused = []
            for path, vec_score in vec_results:
                ks = (kw_scores.get(path, 0.0) / max_kw) if max_kw > 1e-10 else 0.0
                vs = (vec_score / max_vec) if max_vec > 1e-10 else 0.0
                final_score = kw_weight * ks + vec_weight * vs
                fused.append((path, final_score))

            fused.sort(key=lambda x: x[1], reverse=True)
            vec_results = fused[:top]
        else:
            # Pure vector search
            vec_results = search_vectors(
                query_embedding=query_vec,
                store_dir=str(self._store),
                domain=domain or "",
                top_k=top,
            )

        return [
            SearchResult(
                path=path,
                score=score,
                content=(self._store / path).read_text(encoding="utf-8") if (self._store / path).exists() else "",
                domain=path.split("/")[0] if len(path.split("/")) >= 3 else "",
            )
            for path, score in vec_results
        ]

    # ------------------------------------------------------------------
    # Ask (RAG)
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        domain: Optional[str] = None,
        use_llm: bool = True,
    ) -> AskResult:
        """Ask a question using RAG."""
        results = self.search(question, mode="hybrid", domain=domain, top=10)

        if not results:
            return AskResult(answer=None, context="", sources=[])

        # Graph expansion: pull in related/referenced chunks
        edges_file = self._edges_path()
        seen_paths = {r.path for r in results}
        expanded = []
        if edges_file.exists():
            edges_text = edges_file.read_text()
            for r in results[:5]:  # Expand top 5 results
                for line in edges_text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    source, target, etype = parts[0], parts[1], parts[2]
                    if source == r.path and etype in ("related_to", "references"):
                        if target not in seen_paths:
                            target_file = self._store / target
                            if target_file.exists():
                                content = target_file.read_text(encoding="utf-8")
                                expanded.append(SearchResult(
                                    path=target, score=r.score * 0.8,
                                    content=content,
                                    domain=target.split("/")[0] if len(target.split("/")) >= 3 else "",
                                ))
                                seen_paths.add(target)

        all_results = results + expanded

        max_context = int(self._read_config("llm.max_context", "8000"))
        context_parts = []
        sources = []
        tokens_used = 0

        for r in all_results:
            chunk_tokens = int(len(r.content.split()) * 1.3)
            if tokens_used + chunk_tokens > max_context:
                break
            context_parts.append(
                f"--- Source: {r.path} (score: {r.score:.4f}) ---\n{r.content}"
            )
            sources.append(r.path)
            tokens_used += chunk_tokens

        context = "\n\n".join(context_parts)

        if not use_llm:
            return AskResult(answer=None, context=context, sources=sources)

        llm_provider = self._read_config("llm.provider", "none")
        if llm_provider == "none":
            return AskResult(answer=None, context=context, sources=sources)

        sys.path.insert(0, str(self._ragdag_dir))
        from engines.llm import get_answer

        llm_model = self._read_config("llm.model", "gpt-4o-mini")

        answer = get_answer(
            question=question,
            context=context,
            provider=llm_provider,
            model=llm_model,
        )

        return AskResult(answer=answer, context=context, sources=sources)

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

    def trace(self, node_path: str) -> List[dict]:
        """Get provenance chain."""
        edges_file = self._edges_path()
        if not edges_file.exists():
            return []

        chain = []
        current = node_path
        visited = set()

        while current not in visited:
            visited.add(current)
            parent = None

            for line in edges_file.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                source, target, etype = parts[0], parts[1], parts[2]
                if source == current and etype in ("chunked_from", "derived_via"):
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

    def link(self, source: str, target: str, edge_type: str = "references"):
        """Create a manual edge."""
        edges_file = self._edges_path()
        with open(edges_file, "a") as f:
            f.write(f"{source}\t{target}\t{edge_type}\t\n")
