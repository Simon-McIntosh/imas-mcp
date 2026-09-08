#!/usr/bin/env python3
"""Classify how a run wrote ITS DECLARED DELIVERABLE: incrementally, in one
closing generation, or never. Scratch writes are excluded deliberately — an
earlier version counted them and separated nothing, because a node that writes
ten scratch query scripts and no report looks 'incremental' by write count."""
import json, sys, os, glob

D = "/home/ITER/mcintos/.config/reckon/crew/runs"

def declared_paths(run_dir):
    mf = os.path.join(run_dir, "manifest.md")
    paths = []
    if os.path.exists(mf):
        for line in open(mf):
            if line.startswith("orientation_write_paths:"):
                try:
                    paths = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    pass
    return [os.path.basename(p) for p in paths]

def analyse(run_dir):
    stream = os.path.join(run_dir, "stream.jsonl")
    if not os.path.exists(stream):
        return None
    targets = declared_paths(run_dir)
    idx = 0
    hits = []
    with open(stream) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            idx += 1
            # claude-shaped stream: message.content[].tool_use
            msg = r.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict) or c.get("type") != "tool_use":
                        continue
                    inp = c.get("input") or {}
                    blob = " ".join(str(inp.get(k) or "") for k in ("file_path","path","command"))
                    if targets and any(t in blob for t in targets):
                        hits.append(idx)
                continue
            # codex-shaped stream: item.completed with a file_change or command
            # NOTE: a first version handled only the claude shape and reported
            # every codex run as "DELIVERABLE NEVER WRITTEN" -- a false negative,
            # not a finding. Any lane-comparison using this must handle both.
            if r.get("type") in ("item.completed", "item.started"):
                it = r.get("item") or {}
                blobs = []
                if it.get("type") == "file_change":
                    blobs += [str(ch.get("path") or "") for ch in (it.get("changes") or [])]
                blobs.append(str(it.get("command") or ""))
                blobs.append(str(it.get("arguments") or ""))
                blob = " ".join(blobs)
                if targets and any(t in blob for t in targets):
                    hits.append(idx)
    total = idx or 1
    if not hits:
        return {"total": total, "n": 0, "verdict": "DELIVERABLE NEVER WRITTEN"}
    pos = [h/total for h in hits]
    lo, hi = min(pos), max(pos)
    if len(hits) >= 3 and (hi - lo) > 0.25:
        v = "incremental"
    elif lo > 0.80:
        v = "one closing generation"
    else:
        v = f"{len(hits)} write(s), span {lo:.2f}-{hi:.2f}"
    return {"total": total, "n": len(hits), "lo": round(lo,3), "hi": round(hi,3), "verdict": v}

for pat in sys.argv[1:]:
    for rd in sorted(glob.glob(os.path.join(D, pat))):
        res = analyse(rd)
        node = os.path.basename(rd).split("-n-", 1)[-1][:44]
        if res is None:
            print(f"{node:46s} (no stream)")
        else:
            print(f"{node:46s} n={res['n']:>2} lo={res.get('lo','-'):>5} hi={res.get('hi','-'):>5}  {res['verdict']}")
