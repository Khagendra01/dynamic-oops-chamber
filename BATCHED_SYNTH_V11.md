# v1.1 — GPU-batched candidate evaluation during OOPS prefix synthesis

`BatchedPrefixOOPS` replaces the CPU `replay()` calls inside the reference
`PrefixOOPS` search with batched `TorchJobPolicy` episodes on a GPU, while the
search decision code itself stays byte-for-byte the reference code.

## Design

* The walk is unchanged: `_run_lane`, phase budgets, lane budgets, acceptance
  and freezing all come from `oops_prefix.PrefixOOPS`.
* Before a stage runs, `_prefetch_stage` enumerates a **cost-is-one superset**
  of the prefixes that lane run could possibly visit (`_enumerate_lane`: every
  node consumes the smallest possible search cost, so any real node is inside
  the tree) and evaluates the `(prefix, task)` pairs on the device. Results go
  into the reference `_cache`; `_evaluate` only falls back to a CPU `replay`
  when a pair was not prefetched.
* Two prefetch policies (`--merge-waves`):
  * **waves** — one device batch per `(lane phase, task)` wave, short-circuiting
    tasks the walk will not reach; fewer episodes, more batches (16).
  * **merged** — every `(prefix, task)` pair of a stage in one wave; more
    episodes, fewer batches (7–8). Fewer batches wins because a batch has a
    fixed price (the kernel runs to the straggler's horizon).
* The compiled step (`torch.compile(policy.step)`, `dynamic=False`) is built
  **once per process** and shared by every depth, mode and repeat. Batch size
  and program width are graph shapes, so they stay fixed for a run.

## Correctness

| check | scope | result |
| --- | --- | --- |
| `test_batched_v11.py` | interpreter parity, batch vs single, all 10 episode fields | pass (2 tests) |
| full unit suite on the GPU image | 63 baseline + 2 batched | 65 pass, rc=0 |
| `parity_check_v11.py` sweep | 6 904 episodes × 10 fields (success, collision, interventions, steps, fallback/explicit waits, continued, directional, trace, work) | 0 mismatches |
| search-level `R.compare` | frozen programs, every `StageResult` field, `real_replay_calls`, `simulated_steps` at d4/d5/d6 × waves/merged | parity on all 6 runs |

A parity failure aborts the Modal run, so every timed run below is a verified
run.

## Wall-clock results (Modal L40S, `torch 2.7.1+cu126`, dtype float64, B=8192)

Warm compile (inductor volume cached): **83.2 s**; cold compile: 149–153 s.
One-time per process, excluded from every number below.

| depth | mode | reference (CPU) | batched | speedup | episodes | batches |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | waves | 1.227 s | 1.497 s | 0.82× | 12 037 | 16 |
| 4 | merged | 1.227 s | **1.072 s** | **1.15×** | 35 712 | 7 |
| 5 | waves | 1.362 s | 1.530 s | 0.89× | 14 459 | 16 |
| 5 | merged | 1.362 s | **1.128 s** | **1.21×** | 42 806 | 7 |
| 6 | waves | 1.363 s | 1.558 s | 0.88× | 14 981 | 16 |
| 6 | merged | 1.363 s | 1.561 s | 0.87× | 46 910 | 8 |

Median of 3 repeats; frozen programs identical to the reference in every case.

### Where the time goes (d4 merged, 1.072 s)

| bucket | seconds | note |
| --- | --- | --- |
| device step loop (`run`) | 0.777 | 7 batches × ~51 steps × 2.2 ms |
| prefetch enumeration + cache keys | ~0.05 | Python, superset tree + pair list |
| host policy build | 0.059 | numpy rows, memoised program bytes |
| host result read-out | 0.104 | `tolist()` over 7 counters + geometry |
| env batch + program compile | 0.020 | cached `base` templates, cached `compile_program` |
| walk (the reference decision code) | 0.013 | plus ~0.05 s of `train_task` bookkeeping outside prefetch |

### Utilisation (the reason the speedup is modest)

| metric | waves | merged (d4) |
| --- | --- | --- |
| fill = real episodes / row slots | 9–11 % | 62–75 % |
| straggler overrun = executed row-steps / the rows' own steps | 1.6–1.7× | 2.0–2.3× |
| device row-step throughput | ~0.37 µs | ~0.27 µs (1.85 M row-steps in 0.777 s) |
| CPU replay throughput (reference) | ~150 µs per single-trajectory step | |

The GPU is ~500× cheaper per row-step, but it executes 2–3× more row-steps
than the episodes need (padding + running to the slowest row) and evaluates a
**superset** (35 712 prefetched pairs versus 3 124 walk queries and 495 unique
CPU replays at d4). Those two waste factors plus the shared Python walk are
what hold the net gain to ~1.15×.

## Why merged beats waves

A device batch costs `steps_until_slowest_row × cost_per_step(B)` almost
regardless of how many rows are in it. Waves pay that fixed price 16 times for
12 k episodes; merged pays it 7 times for 36 k episodes. Episode waste is free
once the batch is filled; batches are not.

## Batch size sweep (measured, B=16384)

B=16384 halves the merged batch count (7 → 5) but doubles the per-step price
(2.19 ms → 4.10 ms per step at B=8192 → 16384), so `run` rises from 0.78 s to
1.00 s and the search gets *slower*:

| depth | B=8192 merged | B=16384 merged |
| --- | --- | --- |
| 4 | 1.072 s (1.15×) | 1.327 s (0.92×) |
| 5 | 1.128 s (1.21×) | 1.377 s (0.99×) |
| 6 | 1.561 s (0.87×) | 1.710 s (0.81×) |

Waves at B=16384 are far worse (2.69 s, 0.46×): 16 batches of mostly-empty
rows. B=8192 is the better of the two measured shapes.

## Known limits

* `program_width` and `batch_size` are graph shapes: changing either forces a
  recompile (~83–150 s).
* dtype must be float64 for reference parity.
* The prefetch superset grows with `max_depth`, which is why d6 needs an 8th
  batch and lands at 0.87×.
* Headroom: with perfect packing (fill 100 %, overrun 1×) the d4 device step
  loop would be 0.25 s instead of 0.78 s, giving ~1.8×. That needs each row's
  horizon before it runs, i.e. the walk's decisions — not observable up front.

## Reproduce

```bash
modal run modal_oops_v11.py                    # d4/d5/d6 × waves/merged, B=8192
BATCH_SIZE=16384 modal run modal_oops_v11.py   # batch-shape sweep
python -m unittest discover -p 'test_*.py' -q  # 65 tests
python parity_check_v11.py --queries ../reference_queries.json --chunk 1024
python run_batched_synthesis_v11.py --device cuda --merge-waves --max-depth 4
```

Results: `/tmp/opencode/oops_v10/l40s_batched_v11.json`
(`l40s_batched_v11_b16384.json` for the B=16384 sweep).
