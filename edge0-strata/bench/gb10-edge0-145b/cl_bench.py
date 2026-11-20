#!/usr/bin/env python3
"""cl_bench - interleaved A/B arms for the Edge0-145B engine work (2026-11-20).

Unlike tools/dgx_bench.py every arm of a round gets the IDENTICAL prompts (seeded by round and cell, fresh server per
arm so nothing is cached), so at temperature 0 two engines that compute the same bits produce the same text and the
same draft acceptance: the decode figures compare like for like, and the answers' hashes are a parity check.

    python cl_bench.py --config du4.json --arms arms.json --rounds 2 --out cl-sweep1

arms.json: [{"tag": "base", "exe": "", "env": {}}, ...]   (exe "" = the config's)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path.home() / "projects/edge0/edge0-strata"
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel "
         "lattice mirror nectar oasis pepper quarry raven saddle tundra umber vessel willow xenon yellow zephyr "
         "basalt cinder dune elm fjord granite heather inlet jetty kiln lagoon mesa niche opal prairie quill").split()

PARITY = [
    ("code", "Write a Python function `parse_iso8601(s)` that parses an ISO-8601 timestamp such as "
             "2026-11-20T21:50:25+08:00 into a tuple without importing datetime. Include a docstring and three asserts."),
    ("zh", "用繁體中文解釋什麼是推測解碼（speculative decoding），以及為什麼草稿被拒絕時輸出仍然正確。大約兩百字。"),
    ("json", "Output only a JSON object describing a hypothetical web project: fields name, version, scripts (object "
             "with build, test, lint) and dependencies (a list of six objects with name and semver range)."),
    ("edit", "Here is a function:\n\n```python\ndef total(items):\n    t = 0\n    for i in items:\n        t = t + i['price'] * i['qty']\n"
             "    return t\n```\n\nRewrite it to skip items whose qty is zero or negative, round the result to 2 decimals, "
             "and add type hints. Show the full new function, then the same function again with a docstring added."),
]


def word_prompt(target_tokens: int, seed: str) -> str:
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(max(8, int(target_tokens / 1.35))))
    return (f"Request nonce {hashlib.sha1(seed.encode()).hexdigest()[:12]}. Read the word list below carefully.\n\n{body}\n\n"
            f"Write a long, detailed analysis of the word list above: its sounds, its themes and its structure. "
            f"Keep writing for at least 400 words; do not stop early and do not ask questions.")


def post(port: int, prompt: str, max_tokens: int, temperature: float = 0.0, timeout: float = 900) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": temperature, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    out = "".join(text)
    return {"ttft_s": ttft, "wall_s": round(time.time() - t0, 3), "text": out,
            "sha": hashlib.sha1(out.encode()).hexdigest()[:12]}


def timings(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/status", timeout=10) as r:
            return json.loads(r.read()).get("last_timings") or {}
    except Exception:
        return {}


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 900) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def run_one(base_cfg: dict, arm: dict, rnd: int, cells, port: int, out: Path, req_timeout: float,
            parity_max: int = 200, skip_warm: bool = False) -> list[dict]:
    tag = arm["tag"]
    d = out / tag
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_cfg)
    if arm.get("exe"):
        cfg["exe"] = arm["exe"]
    if arm.get("xargs"):
        cfg["args"] = list(cfg.get("args") or []) + list(arm["xargs"])
    cfg["env"] = dict(cfg.get("env") or {}, **(arm.get("cfg_env") or {}))
    cfg["port"] = port
    cfg["log"] = str(d / f"{tag}.r{rnd}.engine.log")
    cpath = d / f"{tag}.config.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    penv = dict(os.environ, **{k: str(v) for k, v in (arm.get("env") or {}).items()})
    srvlog = open(d / f"{tag}.server.log", "a", encoding="utf-8")
    proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(cpath), "--port", str(port)], cwd=str(ROOT), env=penv,
                            stdout=srvlog, stderr=srvlog)
    rows: list[dict] = []
    try:
        ready = wait_ready(port, proc)
        print(f"[r{rnd}] {tag}: ready in {ready:.0f} s", flush=True)
        post(port, f"Request {rnd}. Say hello in one short sentence, then list five colors.", 64, timeout=req_timeout)
        if not skip_warm:
            post(port, word_prompt(4096, f"warm-{rnd}"), 8, timeout=req_timeout)   # pages the experts in (dgx_bench does the same)
        for name, prompt in PARITY:
            r = post(port, prompt, parity_max, timeout=req_timeout)
            t = timings(port)
            rows.append({"round": rnd, "arm": tag, "kind": "parity", "cell": name, **r, **t})
            print(f"    parity {name:5s}: sha {r['sha']} decode {t.get('predicted_per_second')} tok/s "
                  f"drafts {t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
        for (length, maxnew) in cells:
            r = post(port, word_prompt(length, f"cell-{rnd}-{length}-{maxnew}"), maxnew, timeout=req_timeout)
            t = timings(port)
            rows.append({"round": rnd, "arm": tag, "kind": "perf", "cell": f"{length}:{maxnew}", **r, **t})
            print(f"    {length:6d} n={maxnew:4d}: sha {r['sha']} prompt_n {t.get('prompt_n')} prompt "
                  f"{t.get('prompt_per_second')} tok/s decode {t.get('predicted_per_second')} tok/s n {t.get('predicted_n')} "
                  f"ttft {r['ttft_s'] and round(r['ttft_s'], 2)}", flush=True)
    except Exception as e:   # a hang or a crash is a result: keep it and go on with the next arm
        print(f"    {tag}: FAILED: {type(e).__name__}: {e}", flush=True)
        rows.append({"round": rnd, "arm": tag, "kind": "error", "cell": "-", "error": f"{type(e).__name__}: {e}"})
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["pkill", "-KILL", "-f", str(cfg["exe"]) + " --serve"], check=False)
        time.sleep(6)
    with open(d / f"{tag}.jsonl", "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cells", default="4096:512,16384:256")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--out", required=True)
    ap.add_argument("--req-timeout", type=float, default=300)
    ap.add_argument("--parity-max", type=int, default=200, help="new tokens per parity prompt")
    ap.add_argument("--skip-warm", action="store_true", help="no 4K warm-up prompt (logit-dump runs)")
    a = ap.parse_args()
    base_cfg = json.loads(Path(a.config).read_text())
    arms = json.loads(Path(a.arms).read_text())
    cells = [tuple(int(x) for x in c.split(":")) for c in a.cells.split(",") if c]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))   # A B / B A
        for arm in order:
            rows += run_one(base_cfg, arm, rnd, cells, a.port, out, a.req_timeout, a.parity_max, a.skip_warm)
    med = lambda v: statistics.median(v) if v else None
    fmt = lambda v: "-" if v is None else f"{v:.1f}"
    lines = ["| arm | cell | n | prompt tok/s | decode tok/s | drafts acc |", "|---|---|---|---|---|---|"]
    for arm in arms:
        for cell in [f"{l}:{m}" for (l, m) in cells] + ["parity(all)"]:
            sel = [r for r in rows if r["arm"] == arm["tag"] and (r["cell"] == cell or (cell == "parity(all)" and r["kind"] == "parity"))]
            acc = sum(r.get("draft_n_accepted") or 0 for r in sel), sum(r.get("draft_n") or 0 for r in sel)
            lines.append(f"| {arm['tag']} | {cell} | {len(sel)} | "
                         f"{fmt(med([r['prompt_per_second'] for r in sel if r.get('prompt_per_second')])) if cell != 'parity(all)' else '-'} | "
                         f"{fmt(med([r['predicted_per_second'] for r in sel if r.get('predicted_per_second')]))} | {acc[0]}/{acc[1]} |")
    ref = arms[0]["tag"]
    par = []
    for arm in arms[1:]:
        same = diff = 0
        for r in rows:
            if r["arm"] != arm["tag"] or r["kind"] == "error":
                continue
            m = [x for x in rows if x["arm"] == ref and x["round"] == r["round"] and x["cell"] == r["cell"] and x["kind"] == r["kind"]]
            if m:
                same += m[0]["sha"] == r["sha"]
                diff += m[0]["sha"] != r["sha"]
        par.append(f"{arm['tag']} vs {ref}: {same} answers identical, {diff} differ")
    errs = [f"{r['arm']} r{r['round']}: {r['error']}" for r in rows if r["kind"] == "error"]
    summary = "\n".join(lines) + "\n\n" + "\n".join(par) + ("\n\nerrors:\n" + "\n".join(errs) if errs else "") + "\n"
    (out / "summary.md").write_text(summary)
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
