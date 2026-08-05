#!/usr/bin/env python3
"""Verify a downloaded .qpack container against its Hugging Face repo.

Compares the SHA-256 of every large file with the hash Hugging Face stores,
deletes any corrupt file (plus manifest.json so the container reads as
incomplete), and tells you to re-run the installer, which re-downloads only
what is missing.

Usage:
  python3 scripts/verify_container.py <org/repo> <container-dir>

Example:
  python3 scripts/verify_container.py Leonickson/Qwen3.6-35B-A3B-qpack \
      ~/models/qwen3.6-35b.qpack
"""
import hashlib
import json
import os
import sys
import urllib.request


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    repo, root = sys.argv[1], os.path.expanduser(sys.argv[2])

    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)
    expected = {
        f["path"]: f["lfs"]["oid"]
        for f in tree
        if f.get("type") == "file" and f.get("lfs")
    }
    if not expected:
        print("no large files found in the repo listing; nothing to verify")
        return 2

    bad, missing, ok = [], [], 0
    for rel, want in sorted(expected.items()):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() == want:
            ok += 1
            print(f"ok       {rel}")
        else:
            bad.append(rel)
            print(f"CORRUPT  {rel}")

    print(f"\n{ok} ok, {len(bad)} corrupt, {len(missing)} missing "
          f"(of {len(expected)} large files)")
    if bad:
        for rel in bad:
            os.remove(os.path.join(root, rel))
        manifest = os.path.join(root, "manifest.json")
        if os.path.exists(manifest):
            os.remove(manifest)
        print("\nCorrupt files deleted. Re-run the installer to re-download"
              " just those files:")
        print(f"  swiftlet-repack --from-hf {repo} --output {root}")
    elif missing:
        print("\nRe-run the installer to fetch the missing files.")
    else:
        print("\nContainer verified clean.")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
