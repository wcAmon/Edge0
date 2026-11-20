# Edge0-145B NVFP4 on a DGX Spark (GB10): the all-resident latent window, a PLE ordering race, and Q4 dense tables

A record of one evening's engine work on one machine (20 November 2026). One DGX Spark (GB10, 121.7 GiB unified
memory, Ubuntu 24.04, CUDA 13.0), engine 0.1.55 at `43ceac6`, the Edge0-145B NVFP4 pack with every one of its 24,576
experts in the GPU's cache, `--spec 4`, `--kv int8`, `--prefill auto:32768`. Nothing here was measured on another
machine. Scripts: `bench/gb10-edge0-145b/`.

## What changed in the engine

1. **The latent layer of the zero-doorbell (all-resident) window graph** (`src/core/verify.cpp`).
   `STRATA_VERIFY_ALL_RESIDENT=1` hung the first decode window on a latent pack (the stall of #200): the latent
   branch of `post` waited on flags A, B and M per layer, and the host of that graph raises only the PLE flag, once.
   The branch now has its own all-resident form: the device's `resident_plan`, the A stage on the VRAM slots, U for
   every routed row straight into `parts_`. The D stage's four launches that only feed the CPU share (the q8_0
   activation image, two D2H copies, the xl doorbell) are skipped in that graph. The switch stays opt-in.
2. **`latent_u_matvec` leaves early** when every routed row is the GPU's own (`src/kernels/cuda/latent.cu`). It used
   to scan `n_entries x count` per thread to skip each row one by one. Same result.
3. **The PLE rows are gathered before layer 0's flag goes up again** (`STRATA_PLE_BEFORE_FLAG`, now on by default,
   `=0` restores the previous order). See "A race in the doorbell graph" below.

## Decode

Same prompts for every arm of a round (seeded), a fresh server per arm, arms mirrored between rounds (A B / B A),
temperature 0, medians of two rounds, tok/s. Q4 dense tables in every arm.

| Arm | 4K prompt, 512 new | 16K prompt, 256 new | four short prompts (code, zh, JSON, edit) | prompt 4K / 16K |
|---|---:|---:|---:|---:|
| engine as it was | 30.9 | 29.4 | 41.2 | 1,028 / 1,167 |
| `STRATA_SH_STREAM=0` only | 33.9 | 32.4 | 47.0 | 1,028 / 1,171 |
| this branch + `STRATA_VERIFY_ALL_RESIDENT=1` | 37.8 | 35.8 | 50.8 | 1,033 / 1,172 |
| the same + `STRATA_SH_STREAM=0` | 37.5 | 35.5 | 50.0 | 1,035 / 1,168 |

The all-resident window graph has about 1,850 nodes where the doorbell graph had about 2,600, and no host step per
layer. `tools/agent_turns_bench.py` (23 coding-agent turns up to 53K tokens, temperature 1, one round): median decode
35.1 -> 41.4 tok/s, x1.22 per turn (x0.83 to x1.59), 20 of 23 turns faster, no stall, and the graph never fell back
to the doorbell one. Prompt speed did not change beyond the run-to-run spread (about 15% between sessions here).

Where a window's 55.6 ms go afterwards (`STRATA_VERIFY_PROFILE=1`, average 2.9 rows): the mixers' dense products
about 14.5, the A stage 10.0, U and the combine 8.1, the D stage about 5, the head 3.2, the hyper-connection reads
about 6; nothing waits for the host. The profile is consistent with a window bound by memory bandwidth (roughly
170 GB/s over the bytes it reads), not by launches.

## A race in the doorbell graph (the default path)

Two runs of the unchanged engine, same greedy requests, same window sizes (`STRATA_SPEC_COST=0`), compared row by row
with `STRATA_LOGITS_DUMP`: only 105 of 347 rows were bit-identical and the logits were up to 9.0 apart; the
difference starts at the first generated position. In the host loop the flag of layer 0 is raised before the
window's PLE rows are gathered (the "K0-fix" order), and layer 1 copies `h_ple_` with no wait of its own in that
graph, so the copy can take the previous window's rows.

| Compared | Bit-identical rows | Largest logit difference |
|---|---:|---:|
| rows gathered first, run twice | 461 of 461 | 0 |
| rows gathered first vs the all-resident graph | 461 of 461 | 0 |
| all-resident graph, run twice | 461 of 461 | 0 |
| flag first (as it was) vs any of the above | 8 of 136 | 4.4 |

So the new all-resident branch computes exactly what the doorbell graph computes once the rows are in place, and the
default order was not deterministic on this machine. The K0-fix's own note says the earlier order could deadlock on a
blocked gather; that was not seen here in three short server runs of the doorbell graph, which is a small sample.

## Q4 against Q8 latent dense tables

`STRATA_LATENT_DENSE_FMT=4` (`latent_dense_q4.bin`) measured about 10% more decode than Q8 in an earlier sweep of
the unchanged engine. Both arms below ran on the all-resident graph, which is deterministic, so the difference is the
tables'.

- **Logits:** 1,542 rows with identical inputs (greedy answers to 24 fixed prompts plus the task prompts, compared
  until the two arms first emit different tokens). KL(Q8 || Q4): mean 0.0054, median 0.0002, 95th percentile 0.025,
  largest 0.25. The most likely token changed on 28 rows (1.8%); 23 of those were near ties (the two candidates within
  0.15 of probability; on average 0.39 against 0.31).
- **Tasks:** 18 small tasks with a mechanical check (4 tool calls, 5 functions run against asserts, 2 exact
  extractions of 6 amounts from a 40-row ledger, 2 JSON outputs, 2 sums, 1 ordering puzzle, 2 Traditional Chinese
  outputs). Q8 passed 16, Q4 passed 16, the same 16; the two failures are the same two tool-call tasks in both arms
  (the model explored the repository first instead of making the requested call).

This is a small suite: it would show a gross loss and it did not; it cannot show a loss of a few percent on long
agent sessions.

## What did not work

- A U kernel with one block per (group, 8 rows) that decodes each weight row once for all of the group's routes was
  bit-identical to `latent_u_k` and slower at the usual window sizes (0.52x at 10 routes, 1.03x at 30, 1.28x at 60 for
  Q4). Not kept.
- `--ple-io mmap` instead of `direct`: no difference on the agent turns (prompt 1,078 against 1,095 tok/s).
- Prompt speed: the MoE is 58% of a 16K prompt's GPU time (the D stage 18%, gate/up 16%, down and U 21%, combine 3%)
  and the PLE wait 8% (12% at 4K). Nothing here made it faster.

## Not covered

One machine, two rounds at most, no long soak, no more than one chat slot (`"parallel"` 4 and 8 ended in
`NV_ERR_NO_MEMORY` before this work and were not retried), no pack other than Edge0-145B NVFP4.
