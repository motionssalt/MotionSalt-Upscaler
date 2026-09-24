"""Sync legend2008/kzm-cache-v4 -> motionssalt/proteus-v3-pip-cache (missing files only), then verify file-for-file.
Runs on a GitHub Actions runner (large disk, fast HF<->runner link). Requires env HF_TOKEN (write access to DST)."""
import os, sys, time
from huggingface_hub import HfApi, hf_hub_download

SRC = "legend2008/kzm-cache-v4"
DST = "motionssalt/proteus-v3-pip-cache"
api = HfApi(token=os.environ["HF_TOKEN"])

def tree(repo):
    return {f.path: (f.size or 0) for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
            if getattr(f, "size", None) is not None}

api.create_repo(DST, repo_type="dataset", exist_ok=True)
src, dst = tree(SRC), tree(DST)
print(f"[tree] source {len(src)} files {sum(src.values())/1e9:.2f} GB | dest {len(dst)} files {sum(dst.values())/1e9:.2f} GB", flush=True)
missing = {p: s for p, s in src.items() if dst.get(p) != s}
extra = sorted(set(dst) - set(src))
print(f"[diff] missing={len(missing)} extra={len(extra)}", flush=True)
if extra: print("[diff] EXTRA in dest:", extra, flush=True)

for path, size in sorted(missing.items(), key=lambda kv: kv[1]):
    print(f"[sync] downloading {path} ({size/1e9:.3f} GB)...", flush=True)
    t0 = time.time()
    local = hf_hub_download(repo_id=SRC, filename=path, repo_type="dataset")
    print(f"[sync] downloaded in {time.time()-t0:.0f}s; uploading to {DST}...", flush=True)
    for attempt in (1, 2, 3):
        try:
            t1 = time.time()
            api.upload_file(path_or_fileobj=local, path_in_repo=path, repo_id=DST, repo_type="dataset")
            print(f"[sync] uploaded {path} in {time.time()-t1:.0f}s", flush=True)
            break
        except Exception as e:
            print(f"[sync] upload attempt {attempt} failed: {e}", flush=True)
            if attempt == 3: raise
            time.sleep(20)
    os.remove(local)

dst2 = tree(DST)
still_missing = sorted(p for p in src if dst2.get(p) != src[p])
extra2 = sorted(set(dst2) - set(src))
print(f"[verify] dest now {len(dst2)} files {sum(dst2.values())/1e9:.2f} GB | still_missing={still_missing} extra={extra2}", flush=True)
if still_missing or extra2:
    sys.exit("MIGRATION INCOMPLETE")
print("[verify] OK - byte-size identical, file-for-file, across all", len(src), "files", flush=True)
