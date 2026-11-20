#!/usr/bin/env python3
"""cl_quality_cmp - KL and top-1 agreement between two STRATA_LOGITS_DUMP files (reference first).

Rows are compared per request and position for as long as both runs emitted the same greedy token (the inputs are
then identical, so the row is a teacher-forced comparison).  The row where the top-1 tokens differ is counted (its
inputs are still identical) and ends that request's comparison.

    python cl_quality_cmp.py q8.bin q4.bin
"""
import sys

import numpy as np

from cl_logits_cmp import rows


def logsoftmax(x):
    x = x.astype(np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    (ra, _), (rb, _) = rows(a_path), rows(b_path)
    print(f"requests: {len(ra)} / {len(rb)}")
    fa, fb = open(a_path, "rb"), open(b_path, "rb")
    kls, flips, margins_at_flip, runlen = [], 0, [], []
    for a, b in zip(ra, rb):
        n = 0
        for pos in sorted(set(a) & set(b)):
            (oa, nv), (ob, _) = a[pos], b[pos]
            fa.seek(oa); fb.seek(ob)
            xa = np.frombuffer(fa.read(nv * 4), dtype=np.float32)
            xb = np.frombuffer(fb.read(nv * 4), dtype=np.float32)
            la, lb = logsoftmax(xa), logsoftmax(xb)
            kls.append(float((np.exp(la) * (la - lb)).sum()))
            n += 1
            ta, tb = int(np.argmax(xa)), int(np.argmax(xb))
            if ta != tb:
                flips += 1
                p = np.exp(la)
                margins_at_flip.append((float(p[ta]), float(p[tb])))
                break
        runlen.append(n)
    k = np.array(kls)
    print(f"rows compared (identical inputs): {len(k)}")
    print(f"KL(ref || other): mean {k.mean():.4f}, median {np.median(k):.4f}, p95 {np.percentile(k, 95):.4f}, max {k.max():.4f}")
    print(f"top-1 token differs on {flips} of {len(k)} rows ({100.0 * flips / len(k):.1f}%); "
          f"{sum(1 for a, b in zip(ra, rb) if True) - flips} of {min(len(ra), len(rb))} requests never diverged")
    if margins_at_flip:
        near = sum(1 for p, q in margins_at_flip if p - q < 0.15)
        print(f"at the {flips} divergences the reference gave its own token p={np.mean([p for p, _ in margins_at_flip]):.2f} "
              f"and the other arm's token p={np.mean([q for _, q in margins_at_flip]):.2f} on average; {near} were near ties (gap < 0.15)")
    print(f"tokens until divergence per request: median {int(np.median(runlen))}, min {min(runlen)}, max {max(runlen)}")


if __name__ == "__main__":
    main()
