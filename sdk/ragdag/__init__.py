"""ragdag — A knowledge graph engine that runs on flat files and bash."""

from .core import RagDag

__version__ = "1.0.0"


def init(path: str = ".") -> "RagDag":
    """Initialize a new ragdag store and return a RagDag instance."""
    dag = RagDag(path)
    dag._init_store()
    return dag


def open(path: str = ".") -> "RagDag":
    """Open an existing ragdag store."""
    return RagDag(path)
