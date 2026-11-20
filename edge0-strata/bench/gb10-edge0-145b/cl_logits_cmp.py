#!/usr/bin/env python3
"""cl_logits_cmp - compare two STRATA_LOGITS_DUMP files position by position.

A record is {int32 magic 'LOGS', int32 T, int32 n_vocab, int32 record, int64 position} + T * n_vocab float32.
Requests are split where the position falls back.  A window's row t is the logits at position pos0 + t; a row is
valid only if it was not computed past a rejected draft, i.e. its position is below the next window's pos0 (the last
window of a request keeps its first row only).  Two runs are compared per (request, position) for as long as their
greedy tokens agree (after a divergence the inputs differ, so later rows are not comparable).

    python cl_logits_cmp.py a.bin b.bin
"""
import sys

import numpy as np


def windows(path):
    out = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(24)
            if len(hdr) < 24:
                break
            magic, T, nv, rec = np.frombuffer(hdr[:16], dtype=np.int32)
            pos = int(np.frombuffer(hdr[16:], dtype=np.int64)[0])
            assert magic == 0x53474F4C, hex(magic)
            off = f.tell()
            f.seek(int(T) * int(nv) * 4, 1)
            out.append((pos, int(T), int(nv), off))
    return out


def rows(path):
    """[{position: (offset, n_vocab)}] per request, valid rows only."""
    w = windows(path)
    reqs, cur = [], {}
    for i, (pos, T, nv, off) in enumerate(w):
        nxt = w[i + 1][0] if i + 1 < len(w) else None
        last = nxt is None or nxt <= pos          # the request ends with this window
        valid = 1 if last else min(T, nxt - pos)
        for t in range(valid):
            cur[pos + t] = (off + t * nv * 4, nv)
        if last:
            reqs.append(cur)
            cur = {}
    return reqs, len(w)


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    (ra, wa), (rb, wb) = rows(a_path), rows(b_path)
    print(f"{a_path}: {wa} windows, {len(ra)} requests; {b_path}: {wb} windows, {len(rb)} requests")
    fa, fb = open(a_path, "rb"), open(b_path, "rb")
    tot = ident = 0
    worst = 0.0
    for q, (a, b) in enumerate(zip(ra, rb)):
        n = same_bits = 0
        mx = 0.0
        diverged = None
        for pos in sorted(set(a) & set(b)):
            (oa, nv), (ob, _) = a[pos], b[pos]
            fa.seek(oa); fb.seek(ob)
            xa = np.frombuffer(fa.read(nv * 4), dtype=np.float32)
            xb = np.frombuffer(fb.read(nv * 4), dtype=np.float32)
            n += 1
            same_bits += bool(np.array_equal(xa.view(np.uint32), xb.view(np.uint32)))
            fin = np.isfinite(xa) & np.isfinite(xb)
            mx = max(mx, float(np.max(np.abs(xa[fin] - xb[fin]))) if fin.any() else 0.0)
            if int(np.argmax(xa)) != int(np.argmax(xb)):
                diverged = pos
                break
        tot += n; ident += same_bits; worst = max(worst, mx)
        print(f"  request {q}: {len(a)} / {len(b)} valid rows, compared {n}: {same_bits} bit-identical, max |dlogit| {mx:.6f}"
              + (f", greedy tokens diverge at position {diverged}" if diverged is not None else ", greedy tokens agree throughout"))
    print(f"TOTAL: {tot} rows compared, {ident} bit-identical, max |dlogit| {worst:.6f}")


if __name__ == "__main__":
    main()
