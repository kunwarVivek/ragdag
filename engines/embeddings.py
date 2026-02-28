"""Binary embedding storage — read/write embeddings.bin and manifest.tsv."""

import hashlib
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Magic number: "RAGD" = 0x52414744
MAGIC = 0x52414744
FORMAT_VERSION = 1
HEADER_SIZE = 32


def model_hash(model_name: str) -> int:
    """First 4 bytes of SHA256 of model name as uint32."""
    h = hashlib.sha256(model_name.encode()).digest()
    return struct.unpack("I", h[:4])[0]


def write_embeddings(
    output_dir: str,
    vectors: List[List[float]],
    chunk_paths: List[str],
    dimensions: int,
    model_name_str: str,
    append: bool = True,
) -> None:
    """Write vectors to embeddings.bin and chunk paths to manifest.tsv."""
    out = Path(output_dir)
    bin_path = out / "embeddings.bin"
    manifest_path = out / "manifest.tsv"

    existing_vectors = []
    existing_paths = []

    # Load existing data if appending
    if append and bin_path.exists() and manifest_path.exists():
        try:
            existing_vectors_arr, _, _, _ = load_embeddings(str(bin_path))
            existing_vectors = existing_vectors_arr.tolist()
            existing_paths = _read_manifest_paths(str(manifest_path))

            # Remove entries for chunk_paths that are being re-embedded
            new_path_set = set(chunk_paths)
            filtered = [
                (v, p)
                for v, p in zip(existing_vectors, existing_paths)
                if p not in new_path_set
            ]
            if filtered:
                existing_vectors, existing_paths = zip(*filtered)
                existing_vectors = list(existing_vectors)
                existing_paths = list(existing_paths)
            else:
                existing_vectors = []
                existing_paths = []
        except Exception:
            existing_vectors = []
            existing_paths = []

    # Combine
    all_vectors = existing_vectors + vectors
    all_paths = existing_paths + chunk_paths

    if not all_vectors:
        return

    # Write binary file
    count = len(all_vectors)
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
        arr = np.array(all_vectors, dtype=np.float32)
        f.write(arr.tobytes())

    # Write manifest
    with open(manifest_path, "w") as f:
        f.write("# relative_chunk_path\tindex\tbyte_offset\tdimensions\n")
        for i, path in enumerate(all_paths):
            offset = HEADER_SIZE + i * dimensions * 4
            f.write(f"{path}\t{i}\t{offset}\t{dimensions}\n")


def load_embeddings(bin_path: str) -> Tuple[np.ndarray, int, int, int]:
    """Load embeddings from binary file.

    Returns: (vectors_array, dimensions, count, model_hash)
    """
    with open(bin_path, "rb") as f:
        data = f.read(HEADER_SIZE)
        magic = struct.unpack_from("I", data, 0)[0]
        if magic != MAGIC:
            raise ValueError(f"Not a ragdag embeddings file (magic={hex(magic)})")

        version = struct.unpack_from("I", data, 4)[0]
        dims = struct.unpack_from("I", data, 8)[0]
        count = struct.unpack_from("I", data, 12)[0]
        mhash = struct.unpack_from("I", data, 16)[0]

        vec_data = f.read(count * dims * 4)
        vectors = np.frombuffer(vec_data, dtype=np.float32).reshape(count, dims)

    return vectors, dims, count, mhash


def load_embeddings_mmap(bin_path: str) -> Tuple[np.ndarray, int, int, int]:
    """Load embeddings using memory-mapped file for efficiency."""
    import mmap

    with open(bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        magic = struct.unpack_from("I", mm, 0)[0]
        if magic != MAGIC:
            raise ValueError(f"Not a ragdag embeddings file (magic={hex(magic)})")

        dims = struct.unpack_from("I", mm, 8)[0]
        count = struct.unpack_from("I", mm, 12)[0]
        mhash = struct.unpack_from("I", mm, 16)[0]

        vectors = np.frombuffer(
            mm, dtype=np.float32, offset=HEADER_SIZE, count=count * dims
        ).reshape(count, dims)

    return vectors, dims, count, mhash


def _read_manifest_paths(manifest_path: str) -> List[str]:
    """Read chunk paths from manifest.tsv."""
    paths = []
    with open(manifest_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if parts:
                paths.append(parts[0])
    return paths


def load_manifest(manifest_path: str) -> List[Tuple[str, int, int, int]]:
    """Load manifest entries: (path, index, byte_offset, dimensions)."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                entries.append(
                    (parts[0], int(parts[1]), int(parts[2]), int(parts[3]))
                )
    return entries
