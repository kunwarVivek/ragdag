"""Append-only embedding writes — O(1) ingest for new vectors.

When adding NEW paths (not already in the manifest), vectors are appended
to the end of embeddings.bin and the header count is updated in-place,
avoiding the full-rewrite cost of write_embeddings().

When REPLACING existing paths, falls back to write_embeddings() for
correctness.
"""

import struct
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from .embeddings import (
        FORMAT_VERSION,
        HEADER_SIZE,
        MAGIC,
        _content_hash,
        _read_manifest_paths,
        load_embeddings,
        model_hash,
        write_embeddings,
    )
except ImportError:
    from embeddings import (
        FORMAT_VERSION,
        HEADER_SIZE,
        MAGIC,
        _content_hash,
        _read_manifest_paths,
        load_embeddings,
        model_hash,
        write_embeddings,
    )


def needs_rewrite(output_dir: str, new_paths: List[str]) -> bool:
    """Check if any new_paths already exist in the manifest.

    Returns True if a full rewrite is needed (paths overlap with existing),
    False if append-only is safe.
    """
    if not new_paths:
        return False

    manifest_path = Path(output_dir) / "manifest.tsv"
    if not manifest_path.exists():
        return False

    existing_paths = set(_read_manifest_paths(str(manifest_path)))
    return bool(existing_paths.intersection(new_paths))


def append_embeddings(
    output_dir: str,
    vectors: List[List[float]],
    chunk_paths: List[str],
    dimensions: int,
    model_name_str: str,
    chunk_texts: Optional[List[str]] = None,
) -> None:
    """Append-only embedding write.

    - If no existing file: creates header + vectors + manifest (like write_embeddings).
    - If adding NEW paths: seeks to end, appends vectors, updates header count.
    - If REPLACING existing paths: falls back to write_embeddings().
    """
    if not vectors:
        return

    out = Path(output_dir)
    bin_path = out / "embeddings.bin"
    manifest_path = out / "manifest.tsv"

    # Compute content hashes up front
    if chunk_texts is not None:
        new_hashes = [_content_hash(t) for t in chunk_texts]
    else:
        new_hashes = [""] * len(chunk_paths)

    # If replacing existing paths, fall back to full rewrite
    if needs_rewrite(output_dir, chunk_paths):
        write_embeddings(
            output_dir, vectors, chunk_paths, dimensions,
            model_name_str, append=True, chunk_texts=chunk_texts,
        )
        return

    # If no existing file, create from scratch
    if not bin_path.exists():
        count = len(vectors)
        mhash = model_hash(model_name_str)

        with open(bin_path, "wb") as f:
            # Header: 32 bytes
            f.write(struct.pack("I", MAGIC))
            f.write(struct.pack("I", FORMAT_VERSION))
            f.write(struct.pack("I", dimensions))
            f.write(struct.pack("I", count))
            f.write(struct.pack("I", mhash))
            f.write(b"\x00" * 12)  # reserved

            # Vectors
            arr = np.array(vectors, dtype=np.float32)
            f.write(arr.tobytes())

        # Write manifest from scratch
        with open(manifest_path, "w") as f:
            f.write("# relative_chunk_path\tindex\tbyte_offset\tdimensions\tcontent_hash\n")
            for i, path in enumerate(chunk_paths):
                offset = HEADER_SIZE + i * dimensions * 4
                chash = new_hashes[i]
                f.write(f"{path}\t{i}\t{offset}\t{dimensions}\t{chash}\n")
        return

    # --- Append path: file exists, all paths are new ---

    # Read current count from header
    with open(bin_path, "rb") as f:
        header = f.read(HEADER_SIZE)
    old_count = struct.unpack_from("I", header, 12)[0]

    new_count = old_count + len(vectors)

    # Append vectors to end of file
    arr = np.array(vectors, dtype=np.float32)
    with open(bin_path, "r+b") as f:
        # Update count in header (offset 12)
        f.seek(12)
        f.write(struct.pack("I", new_count))
        # Seek to end and append vectors
        f.seek(0, 2)
        f.write(arr.tobytes())

    # Append to manifest (don't rewrite existing lines)
    with open(manifest_path, "a") as f:
        for i, path in enumerate(chunk_paths):
            idx = old_count + i
            offset = HEADER_SIZE + idx * dimensions * 4
            chash = new_hashes[i]
            f.write(f"{path}\t{idx}\t{offset}\t{dimensions}\t{chash}\n")
