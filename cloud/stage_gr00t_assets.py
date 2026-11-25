"""Stage and verify the immutable public GR00T N1.7 assets on NAS."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1, sha256
from http.client import HTTPException
import os
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


BASE = (
    "nvidia/GR00T-N1.7-3B", "373d34e2723d928870157d07dc216a19a624b5bb", "gr00t-n1.7-3b-373d34e",
)
PROCESSOR = (
    "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "qwen3-vl-2b-processor-8964489",
)
METADATA = {
    "config.json": "0c6a8210fc314901ff442698001c41775e5400b2",
    "processor_config.json": "aece333c65884ccb2fde6f7e7907fd25f659fd74",
    "statistics.json": "15fea3895a0eec788853c53484dcb22561008976",
    "embodiment_id.json": "907dd5cd22ee3b17b24146d86090bfee694efd25",
    "model.safetensors.index.json": "4761851d9515098ac203474f61221a26cd3a98b6",
}
PROCESSOR_FILES = {
    "tokenizer.json": "c6cc1014128b19d1fc46b1d30a23e3b1d35db421",
    "tokenizer_config.json": "d3d3763207692c78780f4bf42d4dadf49a5c8012",
    "preprocessor_config.json": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    "video_preprocessor_config.json": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
    "merges.txt": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
    "vocab.json": "4783fe10ac3adce15ac8f358ef5462739852c569",
    "chat_template.json": "e49e75bdda104c8de941eb20e0b3765184fb27bc",
    "config.json": "0cd8c646eb55794594b14fefcd1d3acea5cc000f",
}
SHARDS = {
    "model-00001-of-00002.safetensors": (4_990_519_232, "8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d"),
    "model-00002-of-00002.safetensors": (1_919_980_184, "c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661"),
}
CHUNK = 16 * 1024 * 1024


def git_blob(data):
    return sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_shard(path, size, checksum):
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected shard size: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != checksum:
        raise ValueError(f"SHA-256 mismatch: {path}")
    print(f"Verified SHA-256 {path.name} {checksum}", flush=True)


def request(url, headers, limit, content_range=None):
    for attempt in range(5):
        separator = "&" if "?" in url else "?"
        req = Request(f"{url}{separator}stage_attempt={attempt}", headers=headers)
        try:
            with urlopen(req, timeout=35) as response:
                if content_range is not None and (
                    response.status != 206 or response.headers.get("Content-Range") != content_range
                ):
                    raise ValueError("Unexpected HTTP byte range")
                data = response.read(limit)
                if content_range is not None and len(data) != limit:
                    raise ValueError("Short HTTP byte range")
                return data
        except (OSError, HTTPException, ValueError) as exc:
            if attempt == 4:
                raise RuntimeError(f"Public asset request failed: {url.split('?')[0]} ({type(exc).__name__})") from None


def stage_metadata(directory, endpoint, snapshot, files):
    repo, revision, relative = snapshot
    destination = directory / relative
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        path = destination / name
        if not path.is_file():
            url = f"{endpoint}/{repo}/resolve/{revision}/{name}"
            data = request(url, {}, limit=16 * 1024 * 1024)
            if git_blob(data) != expected:
                raise ValueError(f"Git blob mismatch: {path}")
            temporary = path.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_bytes(data)
            temporary.replace(path)
            print(f"Staged {relative}/{name}", flush=True)
        elif git_blob(path.read_bytes()) != expected:
            raise ValueError(f"Git blob mismatch: {path}")


def fetch_range(file, start, stop, size, url):
    header = f"bytes={start}-{stop - 1}"
    data = request(
        f"{url}?part={start}-{stop - 1}", {"Range": header}, stop - start,
        content_range=f"bytes {start}-{stop - 1}/{size}",
    )
    if len(data) != stop - start or os.pwrite(file, data, start) != len(data):
        raise OSError(f"Short asset chunk at offset {start}")


def stage_shard(path, size, checksum, url, workers):
    if path.is_file():
        verify_shard(path, size, checksum)
        return
    part = path.with_suffix(".ranges.part")
    marks = path.with_suffix(".ranges")
    if not part.is_file() and marks.exists():
        shutil.rmtree(marks)
    marks.mkdir(exist_ok=True)
    with part.open("r+b" if part.exists() else "w+b") as stream:
        stream.truncate(size)
        pending_offsets = [start for start in range(0, size, CHUNK) if not (marks / str(start)).exists()]
        completed = size - sum(min(CHUNK, size - start) for start in pending_offsets)
        print(f"{path.name}: resuming {completed:,}/{size:,} bytes", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_range, stream.fileno(), start, min(start + CHUNK, size), size, url): start
                for start in pending_offsets
            }
            for future in as_completed(futures):
                future.result()
                start = futures[future]
                (marks / str(start)).touch()
                completed += min(CHUNK, size - start)
                if completed == size or completed // CHUNK % 8 == 0:
                    print(f"{path.name}: {completed:,}/{size:,} bytes", flush=True)
        os.fsync(stream.fileno())
    try:
        verify_shard(part, size, checksum)
    except ValueError:
        part.unlink()
        shutil.rmtree(marks)
        raise
    part.replace(path)
    shutil.rmtree(marks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not args.endpoint.startswith("https://") or args.workers < 1:
        parser.error("Expected a HTTPS endpoint and positive worker count")
    endpoint = args.endpoint.rstrip("/")
    if args.operation == "stage":
        import fcntl

        args.directory.mkdir(parents=True, exist_ok=True)
        with (args.directory / ".stage.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stage_metadata(args.directory, endpoint, BASE, METADATA)
            stage_metadata(args.directory, endpoint, PROCESSOR, PROCESSOR_FILES)
            repo, revision, relative = BASE
            for filename, (size, checksum) in SHARDS.items():
                url = f"{endpoint}/{repo}/resolve/{revision}/{filename}"
                stage_shard(args.directory / relative / filename, size, checksum, url, args.workers)
    else:
        for snapshot, files in ((BASE, METADATA), (PROCESSOR, PROCESSOR_FILES)):
            for name, expected in files.items():
                path = args.directory / snapshot[2] / name
                if git_blob(path.read_bytes()) != expected:
                    raise ValueError(f"Git blob mismatch: {path}")
        for filename, (size, checksum) in SHARDS.items():
            verify_shard(args.directory / BASE[2] / filename, size, checksum)
    print(f"GR00T public assets verified at {args.directory}", flush=True)


if __name__ == "__main__":
    main()
