"""Copy missing/mismatched files from the original Proteus V3 pip-cache dataset
into MotionSalt's own HF dataset, then verify the copy file-for-file.

Reads HF_MIGRATION_TOKEN (write-scoped) from the environment (GitHub Actions
encrypted secret). The source dataset is public and read anonymously.
"""
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

SRC = "legend2008/kzm-cache-v4"
DST = "motionssalt/proteus-v3-pip-cache"

token = os.environ.get("HF_MIGRATION_TOKEN")
if not token:
    sys.exit("HF_MIGRATION_TOKEN env var is not set (repo secret missing?)")

api = HfApi()


def listing(repo_id: str, tok: str | None = None) -> dict[str, int | None]:
    """{path: size_bytes} for every file in a dataset repo (LFS-aware)."""
    info = api.dataset_info(repo_id, files_metadata=True, token=tok)
    out: dict[str, int | None] = {}
    for f in info.siblings:
        size = f.size
        if size is None and getattr(f, "lfs", None):
            size = f.lfs.get("size") if isinstance(f.lfs, dict) else getattr(f.lfs, "size", None)
        out[f.rfilename] = size
    return out


src = listing(SRC)
dst = listing(DST, tok=token)
print(f"source: {len(src)} files | dest: {len(dst)} files", flush=True)

todo = [
    n
    for n, s in src.items()
    if n not in dst
    or (s is not None and dst[n] is not None and dst[n] != s)
]
print(f"to sync: {len(todo)} file(s)", flush=True)

for n in todo:
    gb = (src[n] or 0) / 1e9
    print(f"[sync] {n} ({gb:.2f} GB)", flush=True)
    local = hf_hub_download(SRC, n, repo_type="dataset")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=n,
        repo_id=DST,
        repo_type="dataset",
        token=token,
    )
    os.remove(local)
    print(f"[ok]   {n}", flush=True)

# ── verification pass ────────────────────────────────────────────────────────
dst2 = listing(DST, tok=token)
missing = [n for n in src if n not in dst2]
mismatch = [
    n
    for n in src
    if n in dst2 and src[n] is not None and dst2[n] is not None and src[n] != dst2[n]
]
extra = [n for n in dst2 if n not in src]
print(
    f"VERIFY: src={len(src)} dst={len(dst2)} "
    f"missing={len(missing)} size_mismatch={len(mismatch)} extra={len(extra)}",
    flush=True,
)
for n in missing:
    print("MISSING:", n, flush=True)
for n in mismatch:
    print(f"SIZE MISMATCH: {n} src={src[n]} dst={dst2[n]}", flush=True)
for n in extra:
    print("extra (not in source, kept):", n, flush=True)

if missing or mismatch:
    sys.exit("verification FAILED")
print("VERIFY OK — dest matches source file-for-file.", flush=True)
