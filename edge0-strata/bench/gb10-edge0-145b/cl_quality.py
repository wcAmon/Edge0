#!/usr/bin/env python3
"""cl_quality - Q8 vs Q4 latent dense (D/U) tables on Edge0-145B: one server run per arm.

Part A (logits): 24 fixed greedy prompts, 48 new tokens each, with STRATA_LOGITS_DUMP set by the arm; the dumps are
compared afterwards (cl_quality_cmp.py) for as long as both arms emit the same tokens.
Part B (tasks): 18 small tasks with a mechanical check (tool calls, JSON, code that must pass asserts, exact
extraction, arithmetic, required facts in a Chinese summary), temperature 0.

    python cl_quality.py --config du4.json --arm '{"tag":"q4","exe":"...","env":{...}}' --out cl-quality
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cl_bench import ROOT, post, wait_ready  # noqa: E402

KL_PROMPTS = [
    "Explain in plain English how a hash map handles collisions, and name two strategies with one trade-off each.",
    "Write a Python function that merges two sorted lists into one sorted list without using sort(). Add comments.",
    "用繁體中文說明 TCP 三向交握的三個步驟，以及為什麼兩次握手不夠。請寫成一段完整的文字。",
    "List the steps to debug a memory leak in a long-running Node.js service, as a numbered list of short items.",
    "Write a SQL query that returns the top five customers by total order value in 2025, with table and column names.",
    "用繁體中文寫一段給新進工程師的說明：什麼是 code review，以及送出 pull request 之前應該檢查哪些事情。",
    "Describe the difference between a process and a thread to someone who has only written shell scripts so far.",
    "Write a bash script that finds every file larger than 100 MB under a directory given as its first argument.",
    "請用繁體中文比較 REST 與 GraphQL 這兩種 API 設計方式，各舉一個適合使用的情境並說明理由。",
    "Give a JSON object describing a user profile with id, name, email, roles (a list) and settings (an object).",
    "Write a Rust function that returns the nth Fibonacci number iteratively, and explain its time complexity.",
    "Summarize what a database index is, when it speeds a query up, and when it slows writes down, in one paragraph.",
    "用繁體中文解釋什麼是快取失效（cache invalidation），並說明常見的兩種策略與各自的缺點。",
    "Write a TypeScript interface for a paginated API response, then a function that fetches every page in a loop.",
    "Explain what a race condition is using a bank-account example, then show how a mutex prevents it in pseudocode.",
    "Write a Python dataclass for a 2D vector with add, subtract, dot product and length methods, with type hints.",
    "請用繁體中文寫出 git rebase 與 git merge 的差異，並說明在共用分支上使用 rebase 可能造成的問題。",
    "Describe how HTTPS protects a login form in transit: what is encrypted, by which keys, and who verifies whom.",
    "Write a Go function that reads a file line by line and returns the count of lines containing a given substring.",
    "Explain Big-O notation with three everyday examples for O(1), O(n) and O(n log n), one sentence per example.",
    "用繁體中文說明什麼是單元測試、整合測試與端對端測試，並各舉一個網站登入功能的測試例子。",
    "Write a regular expression that matches an ISO date such as 2026-11-20 and explain each part of the pattern.",
    "Give a step-by-step plan for migrating a monolith's user table to a separate service without downtime.",
    "Write a C function that reverses a singly linked list in place and returns the new head, with a short comment.",
]

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository for a regular expression.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace an exact string in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                     "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def chat(port: int, messages, max_tokens=400, tools=None) -> dict:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def text_of(r: dict) -> str:
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def calls_of(r: dict) -> list:
    return ((r.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []


def first_json(s: str):
    s = re.sub(r"^```[a-zA-Z]*\n|```\s*$", "", s.strip(), flags=re.M)
    for i, ch in enumerate(s):
        if ch in "{[":
            for j in range(len(s), i, -1):
                try:
                    return json.loads(s[i:j])
                except ValueError:
                    continue
    return None


def code_of(s: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", s, flags=re.S)
    return m.group(1) if m else s


def run_asserts(code: str, asserts: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(code + "\n\n" + asserts + "\nprint('OK')\n")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], capture_output=True, text=True, timeout=20, cwd=d)
            return r.returncode == 0 and r.stdout.strip().endswith("OK")
        except subprocess.TimeoutExpired:
            return False


def tool_task(user: str, name: str, check):
    def run(port):
        r = chat(port, [{"role": "system", "content": "You are a coding agent. Use the tools; do not answer from memory."},
                        {"role": "user", "content": user}], 200, TOOLS)
        c = calls_of(r)
        if not c:
            return False, text_of(r)[:120]
        f = c[0].get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except ValueError:
            return False, "arguments are not JSON: " + str(f.get("arguments"))[:100]
        return f.get("name") == name and bool(check(args)), f"{f.get('name')} {json.dumps(args, ensure_ascii=False)[:100]}"
    return run


def code_task(user: str, asserts: str):
    def run(port):
        t = text_of(chat(port, [{"role": "user", "content": user + " Reply with one Python code block only."}], 500))
        return run_asserts(code_of(t), asserts), t[:80].replace("\n", " / ")
    return run


def ledger(seed: int, n: int = 40):
    rng = random.Random(seed)
    rows = [(f"INV-{rng.randrange(10000, 99999)}", rng.choice(["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark"]),
             round(rng.uniform(10, 9999), 2)) for _ in range(n)]
    return rows, "\n".join(f"{i} | {c} | {a:.2f}" for i, c, a in rows)


def extract_task(seed: int):
    rows, doc = ledger(seed)
    rng = random.Random(seed + 1)
    want = rng.sample(rows, 6)

    def run(port):
        q = ("Below is an invoice ledger (id | customer | amount).\n\n" + doc + "\n\nReturn only a JSON object mapping each of "
             "these invoice ids to its amount as a number: " + ", ".join(w[0] for w in want) + ".")
        j = first_json(text_of(chat(port, [{"role": "user", "content": q}], 300)))
        ok = isinstance(j, dict) and all(abs(float(j.get(w[0], -1)) - w[2]) < 0.005 for w in want)
        return ok, json.dumps(j)[:100]
    return run


def text_task(user: str, check, max_tokens=300):
    def run(port):
        t = text_of(chat(port, [{"role": "user", "content": user}], max_tokens))
        return bool(check(t)), t[:100].replace("\n", " / ")
    return run


def json_task(user: str, check):
    def run(port):
        j = first_json(text_of(chat(port, [{"role": "user", "content": user}], 400)))
        try:
            return j is not None and bool(check(j)), json.dumps(j, ensure_ascii=False)[:100]
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
    return run


TASKS = [
    ("tool.read", tool_task("Show me what is in src/core/verify.cpp.", "read_file", lambda a: a.get("path") == "src/core/verify.cpp")),
    ("tool.grep", tool_task("Find every place where the function latent_u_grouped is called.", "grep", lambda a: "latent_u_grouped" in a.get("pattern", ""))),
    ("tool.run", tool_task("Run the test suite with pytest in quiet mode.", "run", lambda a: "pytest" in a.get("command", ""))),
    ("tool.edit", tool_task("In config.yaml replace the exact text `port: 8080` with `port: 9090`.", "edit_file",
                            lambda a: a.get("path") == "config.yaml" and a.get("old") == "port: 8080" and a.get("new") == "port: 9090")),
    ("code.rle", code_task("Write `rle(s: str) -> str` that run-length encodes a string: 'aaabcc' -> 'a3b1c2', '' -> ''.",
                           "assert rle('aaabcc') == 'a3b1c2'\nassert rle('') == ''\nassert rle('z') == 'z1'\nassert rle('aabbaa') == 'a2b2a2'")),
    ("code.primes", code_task("Write `primes_below(n: int) -> list[int]` returning all primes below n in increasing order.",
                              "assert primes_below(2) == []\nassert primes_below(20) == [2,3,5,7,11,13,17,19]\nassert len(primes_below(1000)) == 168")),
    ("code.flatten", code_task("Write `flatten(x)` that flattens arbitrarily nested lists into one flat list, keeping order.",
                               "assert flatten([1,[2,[3,[4]],5],[],[[6]]]) == [1,2,3,4,5,6]\nassert flatten([]) == []")),
    ("code.roman", code_task("Write `to_roman(n: int) -> str` for 1 <= n <= 3999.",
                             "assert to_roman(1994) == 'MCMXCIV'\nassert to_roman(4) == 'IV'\nassert to_roman(3999) == 'MMMCMXCIX'\nassert to_roman(58) == 'LVIII'")),
    ("code.lru", code_task("Write a class `LRU` with `__init__(self, cap)`, `get(self, k)` returning -1 when missing, and `put(self, k, v)`; "
                           "it evicts the least recently used key when full.",
                           "c = LRU(2)\nc.put(1,1)\nc.put(2,2)\nassert c.get(1) == 1\nc.put(3,3)\nassert c.get(2) == -1\nc.put(4,4)\n"
                           "assert c.get(1) == -1\nassert c.get(3) == 3\nassert c.get(4) == 4")),
    ("extract.a", extract_task(11)),
    ("extract.b", extract_task(29)),
    ("json.pkg", json_task("Output only JSON: an object with keys name (string), version (semver string) and deps (a list of exactly 4 "
                           "objects, each with name and range).",
                           lambda j: isinstance(j["name"], str) and re.fullmatch(r"\d+\.\d+\.\d+", j["version"]) and len(j["deps"]) == 4
                           and all("name" in d and "range" in d for d in j["deps"]))),
    ("json.sorted", json_task("Output only a JSON array with these numbers sorted ascending: 42, 7, 19, 3, 88, 61, 5, 27, 14, 70.",
                              lambda j: j == [3, 5, 7, 14, 19, 27, 42, 61, 70, 88])),
    ("math.sum", text_task("What is 4817 + 9265 - 1378? Answer with the number only.", lambda t: "12704" in t.replace(",", ""), 30)),
    ("math.mul", text_task("What is 37 * 46? Answer with the number only.", lambda t: "1702" in t.replace(",", ""), 30)),
    ("logic", text_task("Ann is older than Bob. Bob is older than Cid. Dee is younger than Cid. Who is the oldest and who is the "
                        "youngest? Answer as `oldest: X, youngest: Y`.", lambda t: re.search(r"oldest:\s*Ann", t, re.I) and re.search(r"youngest:\s*Dee", t, re.I), 60)),
    ("zh.facts", text_task("請用繁體中文把下面的事實寫成兩句話的摘要，必須保留所有數字與名稱：伺服器 node-7 在 03:14 重啟，原因是記憶體用量達到 97%，"
                           "影響了 1,284 個連線，由值班工程師林怡君處理。",
                           lambda t: all(k in t for k in ("node-7", "03:14", "97", "林怡君")) and ("1,284" in t or "1284" in t))),
    ("zh.translate", text_task("把這句話翻譯成繁體中文，只輸出譯文：The deployment failed because the database migration timed out after 30 seconds.",
                               lambda t: "30" in t and ("資料庫" in t) and ("逾時" in t or "超時" in t), 80)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True, help="JSON: tag, exe, env, cfg_env (merged into the config's env)")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arm = json.loads(a.arm)
    tag = arm["tag"]
    out = Path(a.out).resolve() / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    if arm.get("exe"):
        cfg["exe"] = arm["exe"]
    cfg["env"] = dict(cfg.get("env") or {}, **(arm.get("cfg_env") or {}))
    cfg["port"] = a.port
    cfg["log"] = str(out / f"{tag}.engine.log")
    cpath = out / f"{tag}.config.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    penv = dict(os.environ, **{k: str(v) for k, v in (arm.get("env") or {}).items()})
    srvlog = open(out / f"{tag}.server.log", "a", encoding="utf-8")
    proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(cpath), "--port", str(a.port)], cwd=str(ROOT), env=penv, stdout=srvlog, stderr=srvlog)
    res = {"tag": tag, "kl": [], "tasks": []}
    try:
        print(f"{tag}: ready in {wait_ready(a.port, proc):.0f} s", flush=True)
        for i, p in enumerate(KL_PROMPTS):
            r = post(a.port, p, 48)
            res["kl"].append({"i": i, "sha": r["sha"], "text": r["text"]})
        print(f"{tag}: {len(KL_PROMPTS)} logit prompts done", flush=True)
        for name, fn in TASKS:
            try:
                ok, note = fn(a.port)
            except Exception as e:
                ok, note = False, f"{type(e).__name__}: {e}"
            res["tasks"].append({"task": name, "pass": bool(ok), "note": note})
            print(f"  {name:13s} {'PASS' if ok else 'FAIL'}  {note}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["pkill", "-KILL", "-f", str(cfg["exe"]) + " --serve"], check=False)
        time.sleep(6)
    (out / f"{tag}.quality.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"{tag}: {sum(t['pass'] for t in res['tasks'])} of {len(res['tasks'])} tasks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
