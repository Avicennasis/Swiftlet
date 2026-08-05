#!/usr/bin/env python3
"""Verify a downloaded .qpack container against its source.

Works with two source kinds:
  - a Hugging Face repo (org/repo): hashes come from the HF API
  - any static host / R2 mirror (https URL): hashes come from a
    hashes.json file published next to the container files

Compares the SHA-256 of every file, deletes any corrupt file (plus
manifest.json so the container reads as incomplete), and tells you to
re-run the installer, which re-downloads only what is missing.

Usage:
  python3 scripts/verify_container.py <org/repo | base-url> <container-dir>

Examples:
  python3 scripts/verify_container.py Leonickson/Qwen3.6-35B-A3B-qpack \
      ~/models/qwen3.6-35b.qpack
  python3 scripts/verify_container.py \
      https://pub-c0cfece2dbc340dbb2cd9d94310a7d68.r2.dev/qwen3.6-35b-qpack \
      ~/models/qwen3.6-35b.qpack

Generating hashes.json for a container you are about to mirror:
  python3 scripts/verify_container.py --write-hashes <container-dir>
"""
import hashlib
import json
import os
import sys
import urllib.request


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def local_files(root: str):
    for r, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(r, f), root)
            if not rel.startswith(".") and rel != "hashes.json":
                yield rel


def write_hashes(root: str) -> int:
    out = {}
    for rel in sorted(local_files(root)):
        out[rel] = sha256_of(os.path.join(root, rel))
        print(f"hashed  {rel}")
    with open(os.path.join(root, "hashes.json"), "w") as f:
        json.dump({"files": out}, f, indent=1)
    print(f"\nwrote hashes.json ({len(out)} files)")
    return 0


def expected_hashes(source: str) -> dict:
    if source.startswith("http"):
        base = source.rstrip("/")
        with urllib.request.urlopen(f"{base}/hashes.json", timeout=60) as r:
            return json.load(r)["files"]
    url = f"https://huggingface.co/api/models/{source}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)
    return {
        f["path"]: f["lfs"]["oid"]
        for f in tree
        if f.get("type") == "file" and f.get("lfs")
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--write-hashes":
        return write_hashes(os.path.expanduser(sys.argv[2]))
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, root = sys.argv[1], os.path.expanduser(sys.argv[2])

    expected = expected_hashes(source)
    if not expected:
        print("no files found at the source; nothing to verify")
        return 2

    bad, missing, ok = [], [], 0
    for rel, want in sorted(expected.items()):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        if sha256_of(path) == want:
            ok += 1
            print(f"ok       {rel}")
        else:
            bad.append(rel)
            print(f"CORRUPT  {rel}")

    print(f"\n{ok} ok, {len(bad)} corrupt, {len(missing)} missing "
          f"(of {len(expected)} checked files)")
    if bad:
        for rel in bad:
            os.remove(os.path.join(root, rel))
        manifest = os.path.join(root, "manifest.json")
        if os.path.exists(manifest):
            os.remove(manifest)
        print("\nCorrupt files deleted. Re-run the installer to re-download"
              " just those files.")
    elif missing:
        print("\nRe-run the installer to fetch the missing files.")
    else:
        print("\nContainer verified clean.")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
