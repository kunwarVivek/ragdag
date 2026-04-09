"""Tests for ragdag quality reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag, QualityReport


class TestQualityReportDataclass:
    def test_default_values(self):
        r = QualityReport()
        assert r.total_chunks == 0
        assert r.provenance_coverage == 0.0
        # Empty store: orphan_rate=0, stale_rate=0 => (1-0)+(1-0) = 2 out of 6
        assert abs(r.overall_score - 2.0 / 6.0) < 1e-9

    def test_provenance_coverage(self):
        r = QualityReport(total_chunks=10, chunks_with_provenance=8)
        assert r.provenance_coverage == 0.8

    def test_edge_density(self):
        r = QualityReport(total_documents=5, total_edges=15)
        assert r.edge_density == 3.0

    def test_orphan_rate(self):
        r = QualityReport(total_documents=10, documents_with_edges=7)
        assert r.orphan_rate == 0.3

    def test_stale_rate(self):
        r = QualityReport(synthesis_nodes=10, stale_synthesis_nodes=2)
        assert r.stale_rate == 0.2

    def test_overall_score_perfect(self):
        r = QualityReport(
            total_chunks=10, chunks_with_provenance=10,
            chunks_with_embeddings=10, chunks_in_bm25_index=10,
            total_documents=5, documents_with_edges=5,
            documents_with_synthesis=5, synthesis_nodes=5,
            stale_synthesis_nodes=0,
        )
        assert r.overall_score == 1.0

    def test_summary_output(self):
        r = QualityReport(
            total_chunks=10, chunks_with_provenance=8,
            total_documents=5, total_edges=12,
            edge_types={"chunk_of": 8, "related_to": 4},
            domains=2,
        )
        s = r.summary()
        assert "Quality Report" in s
        assert "80%" in s  # provenance coverage
        assert "chunk_of: 8" in s


class TestQualityMethod:
    def test_quality_empty_store(self, tmp_path):
        dag = ragdag.init(str(tmp_path))
        report = dag.quality()
        assert report.total_chunks == 0
        assert report.total_documents == 0
        assert abs(report.overall_score - 2.0 / 6.0) < 1e-9

    def test_quality_after_ingest(self, tmp_path):
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "readme.md"
        doc.write_text("# Hello\nWorld content here.\n\n# Second\nMore content follows.")
        dag.add(str(doc))

        report = dag.quality()
        assert report.total_chunks >= 2
        assert report.chunks_with_provenance == report.total_chunks  # All new chunks have provenance
        assert report.total_edges >= 1  # chunk_of edges
        assert report.provenance_coverage == 1.0

    def test_quality_with_domain(self, tmp_path):
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "readme.md"
        doc.write_text("# Hello\nWorld.")
        dag.add(str(doc), domain="docs")

        report = dag.quality(domain="docs")
        assert report.domains == 1
        assert report.total_chunks >= 1

    def test_quality_detects_stale_synthesis(self, tmp_path):
        dag = ragdag.init(str(tmp_path))

        # Create a doc dir with a stale synthesis node
        store = tmp_path / ".ragdag"
        domain_dir = store / "docs"
        doc_dir = domain_dir / "readme"
        doc_dir.mkdir(parents=True)
        (doc_dir / "01.txt").write_text("---\ntype: chunk\nsource: /f.md\nheading: \nposition: 1\ntotal: 1\nstrategy: heading\nhash: abc\n---\nContent.")
        (doc_dir / "_summary.txt").write_text("---\ntype: summary\ngenerated: 2026-01-01\nsources: []\nsource_hashes: []\nstale: true\n---\nOld summary.")

        report = dag.quality()
        assert report.synthesis_nodes == 1
        assert report.stale_synthesis_nodes == 1
        assert report.stale_rate == 1.0

    def test_quality_detects_orphan_documents(self, tmp_path):
        dag = ragdag.init(str(tmp_path))

        # Create doc dirs but no edges
        store = tmp_path / ".ragdag"
        for name in ["doc_a", "doc_b"]:
            d = store / "domain" / name
            d.mkdir(parents=True)
            (d / "01.txt").write_text("---\ntype: chunk\nsource: /f.md\nheading: \nposition: 1\ntotal: 1\nstrategy: heading\nhash: abc\n---\nContent.")

        report = dag.quality()
        assert report.total_documents == 2
        assert report.documents_with_edges == 0
        assert report.orphan_rate == 1.0

    def test_quality_summary_is_string(self, tmp_path):
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "readme.md"
        doc.write_text("# Hello\nWorld.")
        dag.add(str(doc))

        report = dag.quality()
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Overall Score" in summary

    def test_quality_backward_compat_bare_chunks(self, tmp_path):
        """Old chunks without provenance are counted but lower the coverage."""
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"
        d = store / "domain" / "old_doc"
        d.mkdir(parents=True)
        (d / "01.txt").write_text("bare chunk without frontmatter")

        report = dag.quality()
        assert report.total_chunks == 1
        assert report.chunks_with_provenance == 0
        assert report.provenance_coverage == 0.0
