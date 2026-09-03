# Benchmark: Qwen3.6-35B-A3B (qpack Q4) on a base Apple M1 Mac mini (16 GB)

Sibling of [`BENCH_QWEN36_35B_M4MAX.md`](BENCH_QWEN36_35B_M4MAX.md): the same
model, prompts, sampler, flags and 1-warmup + 3-measured discipline, run on the
low end of the hardware range from issue #18 — a 2020 Mac mini with the 8-core
M1 GPU and 16 GB of unified memory. Two builds side by side: upstream `main`
at `aaa910a` (which already contains merged #19 + #20) and the PR #24 tip
`5d8b2ab` (S1b layer-major prefill + token-batched GEMVs + S2 Metal attention
+ S3a/S3b instrumentation + the conv scan / attend tiling). Run 2026-09-01
22:47 → 2026-09-02 00:13 EDT on macmini2; 44 processes, every one reported.

The question this run exists to answer is the #18 one: on 16 GB, does the
expert-cache budget (`--cache-gb`) move throughput, and what does each GB cost?

Headline (measured runs only, greedy, 64 new tokens, `--cache-gb 4`):

| Metric | `aaa910a` (upstream) | `5d8b2ab` (PR #24) | delta |
|---|---|---|---|
| Decode, 16-tok prompt | 2.79–2.93 tok/s | **2.91–3.13 tok/s** | +4–8% (like-state pairs, see bimodality note) |
| Decode, 503-tok prompt | 2.22–2.26 tok/s (token-major) | **2.47–2.54 tok/s** (chunk 32); 2.44–2.46 (token-major) | **+12%** / +9% |
| TTFT, 503-tok prompt | 184.4–188.6 s (2.67–2.73 tok/s prefill) | **70.5–72.2 s** (6.97–7.13 tok/s) chunk 32; 157.4–163.8 s (3.07–3.20 tok/s) token-major | **2.6× faster** / −13% |
| TTFT, 16-tok prompt | 5.69–5.92 s | **2.97–3.19 s** | −46% |
| Prefill GPU execution, 503-tok | 75.91–75.97 s | 40.35–40.38 s chunk 32; 75.73–75.90 s token-major | −47% / 0% |
| Command buffers per decode step | 51 (2052 dispatches) | 41 (2082 dispatches) | −10 cb |
| Peak RSS | 6.4–6.8 GiB | 6.2–6.8 GiB | — |
| Swap delta, any run | 0 | 0 | used swap 77.38 MB before and after all 44 runs |
| Greedy output | identical | identical | bitwise equal across builds, cache sizes, schedules, and to the M4 Max logs |

`--cache-gb` sweep, 503-token prompt, decode tok/s (3 measured runs each):

| `--cache-gb` | slots | `5d8b2ab` chunk 32 | `aaa910a` token-major | process footprint | page cache at run start | SSD pageins per run (`5d8b2ab`) |
|---|---|---|---|---|---|---|
| 2 | 1213 | 2.42 / 2.40 / 2.40 | 2.20 / 2.19 / 2.21 | 3.5–3.6 GiB | 9.9–10.1 GiB | 17.4–19.0 GiB |
| 4 | 2427 | 2.52 / 2.54 / 2.47 | 2.24 / 2.26 / 2.22 | 5.5–5.6 GiB | 7.9–8.0 GiB | 13.5–15.2 GiB |
| 6 | 3640 | **2.73 / 2.74 / 2.74** | **2.38 / 2.37 / 2.38** | 7.5–7.6 GiB | 5.9–6.1 GiB | 12.1–13.4 GiB |

Cache size moves decode by +14% (stack) / +8% (baseline) from 2 to 6 GB on
this box (medians of the three measured runs), TTFT by −5% / −13%, and no
budget up to 6 GB swapped. The cost side and the mechanism are in "What this
says for #18" below.

**Follow-up (2026-09-02, `83ce946`):** the CPU gap this run named as the top
lever was sub-attributed and cut — same box, same protocol, `--cache-gb 4`:
decode 3.10–3.11 → 3.85–3.87 tok/s (short prompt, fast page-cache state)
and 2.52 → 3.29 (503-token prompt), TTFT 70.6–72.2 → 58.0–60.9 s, greedy
output still byte-identical to these logs. See "S3c follow-up" at the end.

## Environment

- **Machine**: macmini2 — Mac mini (2020, `Macmini9,1`), Apple M1, 8 CPU cores
  (4 performance + 4 efficiency), **8-core GPU** (`system_profiler
  SPDisplaysDataType`: "Chipset Model: Apple M1 / Total Number of Cores: 8",
  Metal 4), **16 GB unified memory** (`hw.memsize` 17179869184 B), internal
  APFS SSD (`/dev/disk3s5`, 994.7 GB container, 869 GiB free). macOS 26.6.2
  (build 25G83). Swift 6.3.3 (swiftlang-6.3.3.1.3 clang-2100.1.1.101),
  swift-driver 1.148.6, Command Line Tools only (no Xcode).
  `sysctl -n machdep.cpu.brand_string` = `Apple M1`. Swap: 1024 MB backing
  store, 77 MB in use at campaign start (uptime 12 days).
- **Resident daemons** (present in every `ps` idle-check, not stoppable from
  the bench account): WindowServer at a steady ~12% of one CPU core, Grafana
  alloy, wazuh agent, tailscaled. No other swift/clang/make/benchmark
  process during any timed run; `ps aux | sort -nrk3 | head`
  logged into `<run>.pre` before every run plus a soft idle gate (waits for
  any foreign process above 20% CPU).
- **Builds** (fully separated from measurement; both finished before the
  first timed run):
  - baseline `aaa910a5124db3647fa91ef9b37e16091b49e581` (upstream main, "Add
    aggregate Metal step instrumentation (#20)", i.e. #19 + #20 merged)
    exported with `git archive` into `~/build/base-swiftlet`; binary sha256
    `6ba7e741714cddf120c82e36af501b9dacd2473cbb2493a100d68869b0610ce4`.
  - stack `5d8b2ab8c97114d14af4e54a34a60709fdba859c` (PR #24 tip,
    `switftbri/s1b-token-batched-gemv`, 28 commits on top of `aaa910a`,
    clean tree rsync'd into `~/build/s1c-swiftlet`); binary sha256
    `124a35b046a7f19a5a63a0f0b5c6ba1198844335bb75e8d3294f753a7596a20e`.
  - both: `swift build -c release --scratch-path <tree>/.swiftcache`, no
    extra flags, identical `Package.swift`/`Package.resolved`; binaries run
    directly from `.swiftcache/release/swiftlet`.
- **Model**: the same `qwen3.6-35b.qpack` container as the M4 Max runs
  (`hashes.json` pin `1605766290…`, 47 files size-verified), at
  `~/models/qwen3.6-35b.qpack` on the mini, read-only. `hashes.json` sha256
  logged before and after every run (unchanged throughout).

## Protocol

Identical to the M4 Max bench contract (`docs/BENCH_QWEN36_35B_M4MAX.md`,
"Protocol"), restated where the small-RAM box changes anything:

- **Sampler**: greedy argmax (CLI `generate`), EOS ids from the container's
  `config.json`; `--max-new 64`; `--gpu` Metal runtime.
- **Prompts**: the same 16-token fieldfare prompt (`p16.txt`) and the same
  503-token pinned paragraph (`plong.txt`, sha256
  `d5ff9b0a8d0a52935b04d71ad39e874e96176f9ea20f6a105ac5d5ff506fbc19`,
  copied from macbook4's `~/bench-logs/` and re-hashed on the mini before
  the first run). Both tokenize to the same counts as on the M4 Max (16 /
  503 tokens per the `prefill S3a` line of every run).
- **Cache budget**: `--cache-gb 4` is the reference setting on 16 GB
  (2427 slots × 1 769 472 B); the sweep adds 2 (1213 slots) and 6 (3640
  slots). The M4 Max default of 8 was not used: the M4 runs showed 10.9 GiB
  RSS at 8 GB on the long prompt, which leaves no page cache on a 16 GB box.
- **Prefill schedule**: baseline `aaa910a` has no `--prefill-chunk` knob and
  is token-major by construction (one command-buffer chain per prompt
  token); the stack was run token-major (`--prefill-chunk 0`) for the
  like-for-like arm and layer-major chunk 32 (its default) for the headline.
- **Runs**: per cell 1 warmup (`warm-0`) + 3 measured, every measured run
  reported, no best-of; each run a fresh process under `/usr/bin/time -l`.
  Cells were run back-to-back in the order listed in the results; the very
  first process of the campaign (`base-A-c4-warm-0`) is the only one with a
  page cache cold for the campaign, and is labeled so.
- **"Warm" on 16 GB means something different from the M4 doc.** The
  16.88 GiB expert pool cannot live in a 16 GB page cache alongside a 5-9 GiB
  process, so steady state is *partially* warm: each run re-reads whatever
  the previous run evicted from the page cache. Bytes read (misses × stride)
  is therefore an upper bound on SSD traffic (some misses are page-cache
  hits, some are real preads); the S3a/S3b counters do not separate the two.
- **Swap / memory pressure**: `sysctl vm.swapusage` and `vm_stat` captured
  before and after every run (`<run>.pre` / `<run>.post`); the swap column
  below is the used-swap delta across the run. A run that swapped is labeled
  as such, not dropped (reported in the sweep table and the caveats).
- **Thermal**: the mini is a small passively-vented box; `pmset -g therm`
  was logged around every run (it reports no thermal warning level on this
  machine, so throttling is checked by comparing run 1 against run 3 within
  each cell and by the across-cell drift of the repeated `stack-C32-c4`
  configuration).
- **Metric definitions** as in the M4 doc: TTFT = the `prefill S3a` wall
  time (model + tokenizer load excluded; process `real` reported
  separately); decode tok/s as printed over the 64-step loop; peak RSS from
  `/usr/bin/time -l` "maximum resident set size"; bytes read = misses ×
  1 769 472 B; wait = sum of blocking `waitUntilCompleted`; GPU = sum of
  per-command-buffer GPU durations (100% of buffers timed, `err=0`, in
  every run below). S3b phase/buffer split exists only on the stack build
  (`aaa910a` predates S3b; it prints S3a totals and the cache line, which
  are the same code on both sides).
- **Parity**: greedy stdout compared by md5 after stripping the two
  `[QwenMetalModel]` banner lines, across builds, cache sizes, chunk
  settings, and against the M4 Max logs of the same prompts.

Repro (on the mini, `$BIN` = either release binary, `$MODEL` = the qpack):

```sh
$BIN generate $MODEL --gpu --prompt "$(cat p16.txt)"   --max-new 64 --cache-gb {2,4}                      # A (both builds)
$BIN generate $MODEL --gpu --prompt "$(cat plong.txt)" --max-new 64 --cache-gb {2,4,6}                    # B  baseline (token-major by construction)
$BIN generate $MODEL --gpu --prompt "$(cat plong.txt)" --max-new 64 --cache-gb 4 --prefill-chunk 0        # B0 stack, token-major
$BIN generate $MODEL --gpu --prompt "$(cat plong.txt)" --max-new 64 --cache-gb {2,4,6} --prefill-chunk 32 # C  stack, layer-major
```

## Results

Cell labels follow the M4 doc (A = short prompt, B = 503-token token-major,
C = 503-token layer-major chunk 32) with the cache budget suffixed. Cells are
listed in the order they were run. "SSD pageins" is the system-wide `vm_stat`
Pageins delta across the run (16 KiB pages), i.e. real disk reads — it
includes the binary, dylibs and the dense `model.safetensors` when those are
not cached (≤ ~1.5 GiB), and nothing else was running. "Nominal bytes" is the
M4 doc's misses × 1 769 472 B, an upper bound that counts page-cache hits as
if they were reads.

### A — baseline `aaa910a`, 16-token prompt, 64 new, `--cache-gb 4` (token-major by construction)

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cold-0 (warmup, first process of the campaign) | 6.494 s | 2.46 | 3.208 / 2.471 s | 816 / 32 802 | **2.87** | 13.765 / 9.925 s | 20270 / 5330 (79%) | 8.78 GiB | 7.1 GiB | 6.81 GiB | 32.26 s |
| warm-1 | 5.924 s | 2.70 | 3.268 / 2.454 s | 816 / 32 802 | **2.91** | 13.090 / 9.931 s | 20270 / 5330 (79%) | 8.78 GiB | 3.2 GiB | 6.81 GiB | 29.29 s |
| warm-2 | 5.689 s | 2.81 | 3.249 / 2.460 s | 816 / 32 802 | **2.79** | 14.290 / 9.810 s | 20270 / 5330 (79%) | 8.78 GiB | 3.3 GiB | 6.61 GiB | 30.82 s |
| warm-3 | 5.914 s | 2.71 | 3.265 / 2.440 s | 816 / 32 802 | **2.93** | 13.056 / 9.926 s | 20270 / 5330 (79%) | 8.78 GiB | 3.3 GiB | 6.82 GiB | 29.13 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 84 850 / 26 / 35 468 / 42. Measured-run spread: decode 2.79–2.93 tok/s, TTFT 5.689–5.924 s, prefill GPU 2.440–2.460 s, decode GPU 9.810–9.931 s.

### A — stack `5d8b2ab`, 16-token prompt, 64 new, `--cache-gb 4`, layer-major chunk 32 (default)

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 3.124 s | 5.12 | 1.443 / 1.318 s | 41 / 7 479 | **2.92** | 13.815 / 9.897 s | 17152 / 5437 (76%) | 8.96 GiB | 4.0 GiB | 6.47 GiB | 27.37 s |
| warm-1 | 3.160 s | 5.06 | 1.431 / 1.314 s | 41 / 7 479 | **3.13** | 12.407 / 9.723 s | 17152 / 5437 (76%) | 8.96 GiB | 4.1 GiB | 6.82 GiB | 24.93 s |
| warm-2 | 2.972 s | 5.38 | 1.431 / 1.308 s | 41 / 7 479 | **2.91** | 13.944 / 9.874 s | 17152 / 5437 (76%) | 8.96 GiB | 4.1 GiB | 6.59 GiB | 27.26 s |
| warm-3 | 3.186 s | 5.02 | 1.450 / 1.326 s | 41 / 7 479 | **3.09** | 12.588 / 9.860 s | 17152 / 5437 (76%) | 8.96 GiB | 4.1 GiB | 6.82 GiB | 25.23 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 34 325 / 62 / 42 056 / 56. Measured-run spread: decode 2.91–3.13 tok/s, TTFT 2.972–3.186 s, prefill GPU 1.308–1.326 s, decode GPU 9.723–9.874 s.

### A2 — baseline `aaa910a`, 16-token prompt, `--cache-gb 2`

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 5.970 s | 2.68 | 3.252 / 2.449 s | 816 / 32 802 | **2.63** | 13.103 / 10.165 s | 16833 / 8767 (66%) | 14.45 GiB | 0.8 GiB | 4.81 GiB | 32.59 s |
| warm-1 | 5.806 s | 2.76 | 3.341 / 2.448 s | 816 / 32 802 | **2.66** | 13.107 / 10.232 s | 16833 / 8767 (66%) | 14.45 GiB | 0.0 GiB | 4.81 GiB | 31.20 s |
| warm-2 | 5.847 s | 2.74 | 3.350 / 2.450 s | 816 / 32 802 | **2.67** | 13.074 / 10.065 s | 16833 / 8767 (66%) | 14.45 GiB | 0.0 GiB | 4.81 GiB | 31.11 s |
| warm-3 | 5.897 s | 2.71 | 3.380 / 2.449 s | 816 / 32 802 | **2.67** | 13.018 / 10.142 s | 16833 / 8767 (66%) | 14.45 GiB | 0.0 GiB | 4.81 GiB | 31.14 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 42 163 / 20 / 20 / 20. Measured-run spread: decode 2.66–2.67 tok/s, TTFT 5.806–5.897 s, prefill GPU 2.448–2.450 s, decode GPU 10.065–10.232 s.

### A2 — stack `5d8b2ab`, 16-token prompt, `--cache-gb 2`

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 2.325 s | 6.88 | 1.424 / 1.304 s | 41 / 7 479 | **2.80** | 12.690 / 10.033 s | 13773 / 8816 (61%) | 14.53 GiB | 0.4 GiB | 4.82 GiB | 26.46 s |
| warm-1 | 2.347 s | 6.82 | 1.430 / 1.305 s | 41 / 7 479 | **2.81** | 12.752 / 10.157 s | 13773 / 8816 (61%) | 14.53 GiB | 0.0 GiB | 4.82 GiB | 26.46 s |
| warm-2 | 2.368 s | 6.76 | 1.436 / 1.305 s | 41 / 7 479 | **2.81** | 12.655 / 9.976 s | 13773 / 8816 (61%) | 14.53 GiB | 0.0 GiB | 4.82 GiB | 26.41 s |
| warm-3 | 2.323 s | 6.89 | 1.417 / 1.302 s | 41 / 7 479 | **2.89** | 12.489 / 10.052 s | 13773 / 8816 (61%) | 14.53 GiB | 0.1 GiB | 4.82 GiB | 25.79 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 40 / 20 / 20 / 20. Measured-run spread: decode 2.81–2.89 tok/s, TTFT 2.323–2.368 s, prefill GPU 1.302–1.305 s, decode GPU 9.976–10.157 s.

### B — baseline `aaa910a`, 503-token prompt, 64 new, `--cache-gb 4` (token-major by construction)

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 183.899 s | 2.74 | 103.215 / 75.921 s | 25653 / 1 031 152 | **2.24** | 13.141 / 10.438 s | 148791 / 32649 (82%) | 53.80 GiB | 10.3 GiB | 6.81 GiB | 213.76 s |
| warm-1 | 188.466 s | 2.67 | 109.826 / 75.972 s | 25653 / 1 031 152 | **2.24** | 13.713 / 10.493 s | 148791 / 32649 (82%) | 53.80 GiB | 11.3 GiB | 6.43 GiB | 219.36 s |
| warm-2 | 184.449 s | 2.73 | 103.641 / 75.959 s | 25653 / 1 031 152 | **2.26** | 13.093 / 10.486 s | 148791 / 32649 (82%) | 53.80 GiB | 10.3 GiB | 6.81 GiB | 214.14 s |
| warm-3 | 188.589 s | 2.67 | 109.867 / 75.906 s | 25653 / 1 031 152 | **2.22** | 13.881 / 10.482 s | 148791 / 32649 (82%) | 53.80 GiB | 11.2 GiB | 6.44 GiB | 219.73 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 156 / 40 280 / 56 / 38 372. Measured-run spread: decode 2.22–2.26 tok/s, TTFT 184.449–188.589 s, prefill GPU 75.906–75.972 s, decode GPU 10.482–10.493 s.

### B0 — stack `5d8b2ab`, 503-token prompt, `--cache-gb 4`, token-major (`--prefill-chunk 0`)

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 157.452 s | 3.19 | 101.132 / 75.947 s | 20623 / 1 046 242 | **2.43** | 13.166 / 10.399 s | 148791 / 32649 (82%) | 53.80 GiB | 10.4 GiB | 6.81 GiB | 185.15 s |
| warm-1 | 163.818 s | 3.07 | 107.994 / 75.818 s | 20623 / 1 046 242 | **2.45** | 13.714 / 10.428 s | 148791 / 32649 (82%) | 53.80 GiB | 11.3 GiB | 6.42 GiB | 192.26 s |
| warm-2 | 157.354 s | 3.20 | 100.969 / 75.731 s | 20623 / 1 046 242 | **2.46** | 13.074 / 10.431 s | 148791 / 32649 (82%) | 53.80 GiB | 10.4 GiB | 6.81 GiB | 184.83 s |
| warm-3 | 163.642 s | 3.07 | 107.900 / 75.898 s | 20623 / 1 046 242 | **2.44** | 13.732 / 10.342 s | 148791 / 32649 (82%) | 53.80 GiB | 11.3 GiB | 6.40 GiB | 192.04 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 159 / 37 206 / 56 / 38 485. Measured-run spread: decode 2.44–2.46 tok/s, TTFT 157.354–163.818 s, prefill GPU 75.731–75.898 s, decode GPU 10.342–10.431 s.

### C — stack `5d8b2ab`, 503-token prompt, `--cache-gb 4`, layer-major chunk 32 (default)

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 69.448 s | 7.24 | 45.267 / 40.309 s | 656 / 148 809 | **2.52** | 12.856 / 10.375 s | 36452 / 27497 (57%) | 45.31 GiB | 12.2 GiB | 6.06 GiB | 96.17 s |
| warm-1 | 72.210 s | 6.97 | 46.278 / 40.374 s | 656 / 148 809 | **2.52** | 13.549 / 10.246 s | 36452 / 27497 (57%) | 45.31 GiB | 15.2 GiB | 6.25 GiB | 99.85 s |
| warm-2 | 70.505 s | 7.13 | 45.440 / 40.354 s | 656 / 148 809 | **2.54** | 12.787 / 10.298 s | 36452 / 27497 (57%) | 45.31 GiB | 13.5 GiB | 6.61 GiB | 97.06 s |
| warm-3 | 72.106 s | 6.98 | 46.298 / 40.376 s | 656 / 148 809 | **2.47** | 13.728 / 10.388 s | 36452 / 27497 (57%) | 45.31 GiB | 15.2 GiB | 6.27 GiB | 100.32 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 56 / 42 060 / 56 / 40 287. Measured-run spread: decode 2.47–2.54 tok/s, TTFT 70.505–72.210 s, prefill GPU 40.354–40.376 s, decode GPU 10.246–10.388 s.

### C2 — stack `5d8b2ab`, 503-token prompt, chunk 32, `--cache-gb 2`

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 74.359 s | 6.76 | 41.967 / 40.332 s | 656 / 148 809 | **2.41** | 12.580 / 10.437 s | 13888 / 50061 (22%) | 82.50 GiB | 18.0 GiB | 4.81 GiB | 102.20 s |
| warm-1 | 73.555 s | 6.84 | 41.950 / 40.343 s | 656 / 148 809 | **2.42** | 12.541 / 10.407 s | 13888 / 50061 (22%) | 82.50 GiB | 19.0 GiB | 4.91 GiB | 102.17 s |
| warm-2 | 73.362 s | 6.86 | 41.904 / 40.288 s | 656 / 148 809 | **2.40** | 12.658 / 10.421 s | 13888 / 50061 (22%) | 82.50 GiB | 17.4 GiB | 4.81 GiB | 101.33 s |
| warm-3 | 73.734 s | 6.82 | 41.973 / 40.357 s | 656 / 148 809 | **2.40** | 12.643 / 10.401 s | 13888 / 50061 (22%) | 82.50 GiB | 19.0 GiB | 4.91 GiB | 102.53 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 58 / 37 697 / 56 / 37 066. Measured-run spread: decode 2.40–2.42 tok/s, TTFT 73.362–73.734 s, prefill GPU 40.288–40.357 s, decode GPU 10.401–10.421 s.

### C6 — stack `5d8b2ab`, 503-token prompt, chunk 32, `--cache-gb 6`

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 68.848 s | 7.31 | 48.072 / 40.396 s | 656 / 148 809 | **2.74** | 13.513 / 10.037 s | 49573 / 14376 (78%) | 23.69 GiB | 11.2 GiB | 6.51 GiB | 93.73 s |
| warm-1 | 70.075 s | 7.18 | 48.821 / 40.409 s | 656 / 148 809 | **2.73** | 13.656 / 10.063 s | 49573 / 14376 (78%) | 23.69 GiB | 13.2 GiB | 6.45 GiB | 95.62 s |
| warm-2 | 68.999 s | 7.29 | 47.978 / 40.367 s | 656 / 148 809 | **2.74** | 13.523 / 10.100 s | 49573 / 14376 (78%) | 23.69 GiB | 12.1 GiB | 6.63 GiB | 93.86 s |
| warm-3 | 69.910 s | 7.19 | 48.603 / 40.399 s | 656 / 148 809 | **2.74** | 13.507 / 10.029 s | 49573 / 14376 (78%) | 23.69 GiB | 13.4 GiB | 6.44 GiB | 95.74 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 56 / 36 553 / 56 / 42 057. Measured-run spread: decode 2.73–2.74 tok/s, TTFT 68.999–70.075 s, prefill GPU 40.367–40.409 s, decode GPU 10.029–10.100 s.

### B2 — baseline `aaa910a`, 503-token prompt, `--cache-gb 2`

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 206.258 s | 2.44 | 100.549 / 76.853 s | 25653 / 1 031 152 | **2.21** | 12.903 / 10.588 s | 109814 / 71626 (61%) | 118.04 GiB | 10.1 GiB | 4.82 GiB | 236.53 s |
| warm-1 | 204.939 s | 2.45 | 100.383 / 76.774 s | 25653 / 1 031 152 | **2.20** | 12.925 / 10.575 s | 109814 / 71626 (61%) | 118.04 GiB | 9.9 GiB | 4.88 GiB | 236.31 s |
| warm-2 | 205.109 s | 2.45 | 100.493 / 76.883 s | 25653 / 1 031 152 | **2.19** | 12.949 / 10.602 s | 109814 / 71626 (61%) | 118.04 GiB | 8.9 GiB | 4.82 GiB | 235.63 s |
| warm-3 | 204.694 s | 2.46 | 100.158 / 76.669 s | 25653 / 1 031 152 | **2.21** | 12.862 / 10.545 s | 109814 / 71626 (61%) | 118.04 GiB | 10.2 GiB | 4.88 GiB | 235.92 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 173 / 42 050 / 46 / 42 060. Measured-run spread: decode 2.19–2.21 tok/s, TTFT 204.694–205.109 s, prefill GPU 76.669–76.883 s, decode GPU 10.545–10.602 s.

### B6 — baseline `aaa910a`, 503-token prompt, `--cache-gb 6`

| Run | TTFT | Prefill tok/s | Prefill wait / GPU | Prefill cb / dispatches | Decode tok/s | Decode wait / GPU (64 steps) | Hits / misses (rate) | Nominal bytes (misses × stride) | SSD pageins (vm_stat) | Peak RSS | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|
| warm-0 (warmup) | 177.444 s | 2.83 | 111.608 / 75.379 s | 25653 / 1 031 152 | **2.36** | 13.748 / 10.303 s | 165329 / 16111 (91%) | 26.55 GiB | 9.8 GiB | 7.18 GiB | 206.07 s |
| warm-1 | 178.951 s | 2.81 | 112.940 / 75.526 s | 25653 / 1 031 152 | **2.38** | 13.661 / 10.321 s | 165329 / 16111 (91%) | 26.55 GiB | 11.7 GiB | 6.56 GiB | 208.27 s |
| warm-2 | 177.557 s | 2.83 | 111.313 / 75.428 s | 25653 / 1 031 152 | **2.37** | 13.808 / 10.285 s | 165329 / 16111 (91%) | 26.55 GiB | 10.7 GiB | 7.42 GiB | 206.12 s |
| warm-3 | 179.009 s | 2.81 | 113.080 / 75.564 s | 25653 / 1 031 152 | **2.38** | 13.703 / 10.345 s | 165329 / 16111 (91%) | 26.55 GiB | 11.7 GiB | 6.56 GiB | 208.36 s |

Major page faults per run (`/usr/bin/time -l` "page faults"): 56 / 42 052 / 56 / 42 056. Measured-run spread: decode 2.37–2.38 tok/s, TTFT 177.557–179.009 s, prefill GPU 75.428–75.564 s, decode GPU 10.285–10.345 s.

## `--cache-gb` sweep (the #18 table)

Measured runs only; the 2 / 4 / 6 GB rows of each build share prompt, schedule, binary and evening. Slots = ⌊budget × 2³⁰ / 1 769 472⌋. "Page cache at run start" = `vm_stat` file-backed pages in the `warm-1` `.pre` snapshot; footprint = `time -l` "peak memory footprint" (max of the 3 runs).

| Build / schedule | `--cache-gb` | slots | Decode tok/s (3 runs) | TTFT (3 runs) | Hits / misses (rate) | Nominal bytes | SSD pageins per run | Page cache at run start | Peak RSS | Peak footprint | Swap delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stack, chunk 32 | 2 | 1213 | **2.42 / 2.40 / 2.40** | 73.6 s / 73.4 s / 73.7 s | 13888 / 50061 (22%) | 82.5 GiB | 17.4–19.0 GiB | 9.9 GiB | 4.81–4.91 GiB | 3.58 GiB | 0 (77.38 → 77.38 MB) |
| stack, chunk 32 | 4 | 2427 | **2.52 / 2.54 / 2.47** | 72.2 s / 70.5 s / 72.1 s | 36452 / 27497 (57%) | 45.3 GiB | 13.5–15.2 GiB | 7.9 GiB | 6.25–6.61 GiB | 5.59 GiB | 0 (77.38 → 77.38 MB) |
| stack, chunk 32 | 6 | 3640 | **2.73 / 2.74 / 2.74** | 70.1 s / 69.0 s / 69.9 s | 49573 / 14376 (78%) | 23.7 GiB | 12.1–13.4 GiB | 5.9 GiB | 6.44–6.63 GiB | 7.59 GiB | 0 (77.38 → 77.38 MB) |
| baseline, token-major | 2 | 1213 | **2.20 / 2.19 / 2.21** | 204.9 s / 205.1 s / 204.7 s | 109814 / 71626 (61%) | 118.0 GiB | 8.9–10.2 GiB | 10.1 GiB | 4.82–4.88 GiB | 3.53 GiB | 0 (77.38 → 77.38 MB) |
| baseline, token-major | 4 | 2427 | **2.24 / 2.26 / 2.22** | 188.5 s / 184.4 s / 188.6 s | 148791 / 32649 (82%) | 53.8 GiB | 10.3–11.3 GiB | 8.0 GiB | 6.43–6.81 GiB | 5.53 GiB | 0 (77.38 → 77.38 MB) |
| baseline, token-major | 6 | 3640 | **2.38 / 2.37 / 2.38** | 179.0 s / 177.6 s / 179.0 s | 165329 / 16111 (91%) | 26.6 GiB | 10.7–11.7 GiB | 6.1 GiB | 6.56–7.42 GiB | 7.53 GiB | 0 (77.38 → 77.38 MB) |
| stack, 16-tok prompt | 2 | 1213 | **2.81 / 2.81 / 2.89** | 2.3 s / 2.4 s / 2.3 s | 13773 / 8816 (61%) | 14.5 GiB | 0.0–0.1 GiB | 10.0 GiB | 4.82–4.82 GiB | 3.47 GiB | 0 (77.38 → 77.38 MB) |
| stack, 16-tok prompt | 4 | 2427 | **3.13 / 2.91 / 3.09** | 3.2 s / 3.0 s / 3.2 s | 17152 / 5437 (76%) | 9.0 GiB | 4.1–4.1 GiB | 8.0 GiB | 6.59–6.82 GiB | 5.47 GiB | 0 (77.38 → 77.38 MB) |
| baseline, 16-tok prompt | 2 | 1213 | **2.66 / 2.67 / 2.67** | 5.8 s / 5.8 s / 5.9 s | 16833 / 8767 (66%) | 14.4 GiB | 0.0–0.0 GiB | 9.6 GiB | 4.81–4.81 GiB | 3.45 GiB | 0 (77.38 → 77.38 MB) |
| baseline, 16-tok prompt | 4 | 2427 | **2.91 / 2.79 / 2.93** | 5.9 s / 5.7 s / 5.9 s | 20270 / 5330 (79%) | 8.8 GiB | 3.2–3.3 GiB | 8.1 GiB | 6.61–6.82 GiB | 5.46 GiB | 0 (77.38 → 77.38 MB) |

## S3a / S3b breakdowns

S3b on the M1 reports the same counter capability as the M4 Max
(`timestampSet=yes stage=yes(sample resolved) dispatch=no`): per-phase GPU
split is honestly unsupported, encode-side phase times are exact, wait/GPU are
per command-buffer label. The baseline prints S3a totals only (no encode
split), so its "CPU gap" below also contains its encode time — on the stack
that is 4–6 ms per decode step and 0.1–3 s per prefill, so the baseline gap
is overstated by about that much relative to the stack's.

### Decode step decomposition (64-step totals ÷ 64; wall = 1 / tok/s)

CPU gap = wall − wait − encode: everything the CPU thread does between
command buffers — expert-cache misses (pread + copy into the cache slot),
per-layer router readback and top-k, the 248 320-wide argmax, the Swift loop.

| Configuration (warm-1) | tok/s | wall / step | wait | GPU | encode | CPU gap (share) | cb / step | (wait − GPU) per cb |
|---|---|---|---|---|---|---|---|---|
| A baseline, 16-tok, 4 GB | 2.91 | 343.6 ms | 204.5 ms | 155.2 ms | n/a (S3a only) | **139.1 ms (40%)** | 51 | 0.97 ms |
| A stack, 16-tok, 4 GB | 3.13 | 319.5 ms | 193.9 ms | 151.9 ms | 5.4 ms | **120.2 ms (38%)** | 41 | 1.02 ms |
| A2 baseline, 16-tok, 2 GB | 2.66 | 375.9 ms | 204.8 ms | 159.9 ms | n/a (S3a only) | **171.1 ms (46%)** | 51 | 0.88 ms |
| A2 stack, 16-tok, 2 GB | 2.81 | 355.9 ms | 199.2 ms | 158.7 ms | 6.3 ms | **150.4 ms (42%)** | 41 | 0.99 ms |
| B2 baseline, 503-tok, 2 GB | 2.20 | 454.5 ms | 202.0 ms | 165.2 ms | n/a (S3a only) | **252.6 ms (56%)** | 51 | 0.72 ms |
| B baseline, 503-tok, 4 GB | 2.24 | 446.4 ms | 214.3 ms | 164.0 ms | n/a (S3a only) | **232.2 ms (52%)** | 51 | 0.99 ms |
| B6 baseline, 503-tok, 6 GB | 2.38 | 420.2 ms | 213.5 ms | 161.3 ms | n/a (S3a only) | **206.7 ms (49%)** | 51 | 1.02 ms |
| B0 stack token-major, 503-tok, 4 GB | 2.45 | 408.2 ms | 214.3 ms | 162.9 ms | 4.2 ms | **189.7 ms (46%)** | 41 | 1.25 ms |
| C2 stack chunk 32, 503-tok, 2 GB | 2.42 | 413.2 ms | 196.0 ms | 162.6 ms | 4.6 ms | **212.6 ms (51%)** | 41 | 0.81 ms |
| C stack chunk 32, 503-tok, 4 GB | 2.52 | 396.8 ms | 211.7 ms | 160.1 ms | 4.4 ms | **180.7 ms (46%)** | 41 | 1.26 ms |
| C6 stack chunk 32, 503-tok, 6 GB | 2.73 | 366.3 ms | 213.4 ms | 157.2 ms | 4.8 ms | **148.1 ms (40%)** | 41 | 1.37 ms |

On the M4 Max (`3dfd004`) the same gap was 9.6 ms of a 37.9 ms step (25%)
short-context and 15.6 of 45.7 ms (34%) long-context. Here it is **38–56% of
the step** and it
is the only component that moves with the cache budget: stack chunk 32 at
2 / 4 / 6 GB has GPU 162.6 / 160.1 / 157.2 ms and wait 196.0 / 211.7 /
213.4 ms per step, while the gap goes 212.6 → 180.7 → 148.1 ms and the step
413 → 397 → 366 ms. Dispatch latency is also ~5× the M4's: (wait − GPU) per
command buffer is 0.7–1.4 ms on the M1 (0.2 ms on the M4 Max), 41 buffers per
step = 34–56 ms of a decode step that is pure scheduling.

Stack decode S3b (C, warm-1, `--cache-gb 4`, 64 steps): encode attention
0.017 s / delta 0.064 s / moe 0.190 s / router 0.011 s / lmHead 0.001 s;
buffers attention+moe+router cb=640 wait 3.044 s gpu 2.181 s |
delta+moe+router cb=1856 wait 8.993 s gpu 6.664 s | delta+router cb=64 wait
0.122 s gpu 0.081 s | moe+lmHead cb=64 wait 1.389 s gpu 1.320 s. The
per-buffer shape matches the M4 run (same 640/1856/64/64 buffer counts and
5760/24960/97280/5120/128 dispatch counts); only the durations scale.

### Prefill decomposition (whole prefill, warm-1)

| Configuration (warm-1) | TTFT | wait | GPU | encode | CPU gap (share) | cb | dispatches | (wait − GPU) per cb |
|---|---|---|---|---|---|---|---|---|
| A baseline, 16-tok, 4 GB | 5.924 s | 3.268 s | 2.454 s | n/a | **2.66 s (45%)** | 816 | 32 802 | 1.00 ms |
| A stack, 16-tok, 4 GB, chunk 32 | 3.160 s | 1.431 s | 1.314 s | 0.006 s | **1.72 s (55%)** | 41 | 7 479 | 2.85 ms |
| B2 baseline token-major, 2 GB | 204.939 s | 100.383 s | 76.774 s | n/a | **104.56 s (51%)** | 25 653 | 1 031 152 | 0.92 ms |
| B baseline token-major, 4 GB | 188.466 s | 109.826 s | 75.972 s | n/a | **78.64 s (42%)** | 25 653 | 1 031 152 | 1.32 ms |
| B6 baseline token-major, 6 GB | 178.951 s | 112.940 s | 75.526 s | n/a | **66.01 s (37%)** | 25 653 | 1 031 152 | 1.46 ms |
| B0 stack token-major, 4 GB | 163.818 s | 107.994 s | 75.818 s | 3.059 s | **52.77 s (32%)** | 20 623 | 1 046 242 | 1.56 ms |
| C2 stack chunk 32, 2 GB | 73.555 s | 41.950 s | 40.343 s | 0.125 s | **31.48 s (43%)** | 656 | 148 809 | 2.45 ms |
| C stack chunk 32, 4 GB | 72.210 s | 46.278 s | 40.374 s | 0.189 s | **25.74 s (36%)** | 656 | 148 809 | 9.00 ms |
| C6 stack chunk 32, 6 GB | 70.075 s | 48.821 s | 40.409 s | 0.259 s | **21.00 s (30%)** | 656 | 148 809 | 12.82 ms |

Reading the prefill table:

- **Token-major prefill is GPU-identical on both builds** (75.97 vs 75.82 s
  of GPU execution for the same 503 tokens) even though the stack's total
  includes the attention it moved onto the GPU. The attend-tiling GPU gain
  that cut the M4 Max's token-major prefill GPU time by 40% (15.29 → 9.11 s)
  does not show on the 8-core M1's token-major arm. The stack's 25 s
  token-major TTFT gain is on the CPU side: 51 → 41 command buffers per
  token (10 fewer — one per GQA layer, matching S2's GPU attention with a
  GPU-resident KV cache replacing the per-layer round trip) and a CPU gap of
  52.8 vs 78.6 s.
- **Layer-major chunk 32 halves GPU execution** (75.8 → 40.4 s) at 656 vs
  20 623 command buffers and 148 809 vs 1 046 242 dispatches, which is where
  the 2.6× TTFT comes from. Stack chunk-32 S3b (C, warm-1): encode attention
  0.003 s / delta 0.010 s / moe 0.174 s / router 0.002 s; buffers
  attention+moe+router cb=160 wait 10.642 s gpu 9.325 s | delta+moe+router
  cb=464 wait 34.363 s gpu 29.925 s | delta+router cb=16 wait 0.648 s gpu
  0.621 s | moe cb=15 wait 0.572 s gpu 0.466 s | moe+lmHead cb=1 wait
  0.053 s gpu 0.038 s.
- **Where the cache budget goes during prefill**: with chunk 32, GPU time is
  40.3–40.4 s at every budget; the CPU gap shrinks 31.5 → 25.7 → 21.0 s
  (2 → 4 → 6 GB) while (wait − GPU) grows 1.6 → 5.9 → 8.4 s, so TTFT only
  moves 73.6 → 72.2 → 70.1 s. The traces cannot attribute the growth in
  wait − GPU (no per-phase split, no storage timer); it is reported, not
  explained.

## Output parity (production scale)

Same mechanism as the M4 doc — greedy token/text equality, not a logit diff.
Stdout compared by md5 after stripping the two `[QwenMetalModel]` banner
lines:

- **16-token prompt**: all 16 processes (both builds × cache 2/4 × 4 runs)
  produce the identical 64-token continuation (`63de7116…`), and it is
  byte-identical to macbook4's `69820f7` and `3dfd004` A-cell logs.
- **503-token prompt**: all 28 processes (baseline at 2/4/6 GB, stack
  token-major at 4 GB, stack chunk 32 at 2/4/6 GB, 4 runs each) produce the
  identical 64-token continuation (`5f2821bc…`), byte-identical to
  macbook4's B/C/D logs at both `69820f7` and `3dfd004`.
- No divergence at any token position between builds, between token-major
  and layer-major, between cache budgets, or between the M1 and the M4 Max.
- Cache counters are bit-identical run to run within every cell, and the
  token-major counters (148 791 / 32 649 at 4 GB) are identical between the
  two builds — same routing on the same schedule. (Hit *rates* are not
  comparable across schedules: layer-major looks up the expert union once
  per chunk, so it makes 63 949 lookups where token-major makes 181 440;
  misses and bytes are comparable.)
- The `--ids` parity arm was not repeated; on the M4 Max it produced one
  token before EOS, and the 64-token natural-prompt arms above carry the
  sequence-level claim.

## Thermal / drift check

`pmset -g therm` reports "No thermal warning level has been recorded" before
and after every run on this machine (it exposes no CPU speed limit), so
throttling is checked by drift:

- Run 1 vs run 3 inside each cell: prefill GPU execution differs by ≤ 0.1%
  (B: 75.972 vs 75.906 s; C: 40.374 vs 40.376 s; C6: 40.409 vs 40.399 s),
  decode GPU by ≤ 1.4%, decode tok/s by ≤ 2% (C: 2.52 vs 2.47; B: 2.24 vs
  2.22; C6: 2.73 vs 2.74).
- Across the 86-minute campaign the same GPU work drifted the other way: the
  baseline's token-major prefill GPU time was 75.91–75.97 s at 23:10 (B) and
  75.43–75.56 s at 00:05 (B6, the last cell); the stack's chunk-32 prefill
  GPU was 40.35–40.38 s at 22:58 (C) and 40.37–40.41 s at 23:50 (C6).
- No evidence of throttling. The visible run-to-run structure is not
  thermal (next section).

## Caveats

- **Runs alternate between two page-cache states.** In every cell with
  `--cache-gb ≥ 4`, `/usr/bin/time -l` "page faults" (major faults)
  alternates between ~50 and 35 000–42 000 (≈ 0.6 GiB at 16 KiB/page) on
  consecutive runs, and the faulting runs are the slower ones: B 214.1 vs
  219.4–219.7 s real (wait 103.6 vs 109.8 s, GPU 75.9 s in all four); A
  stack 3.13 / 3.09 (fast state) vs 2.91 / 2.92 tok/s (faulting state); A
  baseline 2.91 / 2.93 vs 2.79. The 2 GB cells show no alternation (20 major
  faults per run) and correspondingly tight spreads (2.66–2.67, 2.19–2.21).
  The build-vs-build comparisons above are therefore quoted as ranges, and
  the short-prompt delta as like-state pairs. Mechanism not established from
  these traces; the observation is consistent with each run's expert preads
  evicting file pages the next run then faults back in, which is an
  inference, not a measurement.
- **Steady state is partially cold.** The 16.88 GiB expert pool never fits
  the page cache next to a 3.5–7.6 GiB process: file-backed pages at run
  start were 9.9–10.1 / 7.9–8.1 / 5.9–6.1 GiB at 2 / 4 / 6 GB, and each
  503-token run paged in 9–19 GiB from SSD. "Warm" here means "after a
  warmup run", not "in RAM". The M4 doc's cold-vs-warm distinction does not
  transfer; there was no `sudo purge` (no sudo from the bench account) and
  the first process of the campaign (A baseline `cold-0`) started with
  12.2 GiB of file-backed pages already present from the container's hash
  verification and paged in 7.1 GiB.
- **Memory pressure without swap.** Used swap was 77.38 MB before and after
  all 44 runs, `time -l` recorded 0 swaps in every run, pageouts were ≤ 140
  pages (≤ 2.2 MiB) per run. The compressor did engage: `vm_stat` "Pages
  occupied by compressor" was 5.7–6.8 k pages (~100 MB) at every run start
  but read 83 931 / 105 454 / 177 397 pages (1.3 / 1.6 / 2.7 GiB) in the
  post-run snapshot of C warm-1, B0 warm-3 and C6 warm-1 respectively,
  back to baseline before the next run. Those three runs are not outliers
  in wall time (99.85 vs 97.06–100.32 s; 192.04 vs 184.83–192.26 s; 95.62
  vs 93.86–95.74 s). Peak footprint (`time -l`) exceeds peak RSS at 6 GB
  (7.59 vs 6.44–6.63 GiB), i.e. part of the process was compressed or not
  resident at its peak. `--cache-gb 8` (the CLI default) was not run.
- **Resident load.** WindowServer sits at a steady ~12% of one CPU core on
  this mini (headless, no user session) along with Grafana alloy, wazuh and
  tailscaled; no other swift/clang/benchmark process appeared in any of the
  44 `ps` idle-checks (`<run>.pre`).
- Peak RSS at 6 GB (6.4–7.4 GiB) is not "4 GB RSS + 2 GB": the kernel was
  reclaiming pages from the process at that budget; footprint is the
  better memory number on this box.
- Per-phase GPU split is unsupported on the M1 exactly as on the M4 Max; the
  CPU-gap decomposition is arithmetic and an upper bound that mixes expert
  fetches with router readback and sampling; there is still no dedicated
  storage timer in `StepMetrics`, so fetch cost is isolated by the cache
  differential only.
- SSD pageins are a system-wide counter over the run window on an otherwise
  idle box; they include the process's own binary/dylib/dense-weight faults.
- Single machine, single evening; builds separated from measurement (both
  binaries built before the first timed run: stack 792 s compile including
  the first package fetch, baseline 114 s with packages cached).
- Raw logs (`<cell>-warm-N.{out,err,pre,post}`, `campaign.log`,
  `build.log`, the prompt files, `runcell.sh`, `campaign.sh`) are on
  macmini2 in `~/bench-logs/`.

## What this says for #18

Numbers only, from the tables above:

1. **On a base M1 with 16 GB, `--cache-gb` is a throughput knob, modestly,
   on both builds.** 503-token prompt, decode (medians of the three measured
   runs): stack chunk 32 2.40 → 2.52 → 2.74 tok/s at 2 → 4 → 6 GB (+5.0%,
   then +8.7%; +14.2% end to end); baseline 2.20 → 2.24 → 2.38 (+1.8%, then
   +6.3%; +8.2% end to end). 16-token prompt, 2 → 4 GB: baseline 2.66–2.67
   → 2.79–2.93 (+4–10%), stack 2.81–2.89 → 2.91–3.13 (+1–11%, bimodal).
   TTFT moves less: −5.0% (stack chunk 32) and −12.7% (baseline) from 2 to
   6 GB. The README's #18 note that "`--cache-gb 2` matches `--cache-gb 8`"
   on this class of machine was measured at `f44e29f`; at `aaa910a` and
   `5d8b2ab` the 2-vs-4 decode gap is +2% (baseline, long prompt) to +10%
   (stack, short prompt) and the 2-vs-6 gap is +8% (baseline) to +14%
   (stack).
2. **Each GB of expert cache is taken from the page cache.** Footprint
   3.5–3.6 / 5.5–5.6 / 7.5–7.6 GiB and file-backed pages at run start
   9.9–10.1 / 7.9–8.0 / 5.9–6.1 GiB sum to ~13.5 GiB at every budget. Real SSD reads per 503-token run do not
   fall with the miss count: stack 17.4–19.0 / 13.5–15.2 / 12.1–13.4 GiB
   against 82.5 / 45.3 / 23.7 GiB of nominal misses; baseline 8.9–10.2 /
   10.3–11.3 / 10.7–11.7 GiB against 118.0 / 53.8 / 26.6 GiB — the baseline
   reads *more* from disk at 6 GB than at 2 GB while decoding 8% faster,
   because an in-process hit costs nothing and a page-cache hit still costs
   a 1.7 MiB copy. No budget up to 6 GB swapped; 8 GB was not tested.
3. **What moves is the CPU gap, and it is the largest component of a step
   here.** Wall − wait − encode is 38–56% of a decode step on the M1
   (22–34% on the M4 Max) and is the only term that changes with the cache
   budget (stack chunk 32: 213 → 181 → 148 ms of a 413 → 397 → 366 ms step;
   GPU 163 → 160 → 157 ms; wait 196 → 212 → 213 ms). The wall differential
   the M4 doc used to bound the marginal fetch cost (F − C: 6.8 ms, 9.5% of
   a step, for 8 → 20 GB) is 47 ms, 11% of a step, here for 2 → 6 GB — a
   64 ms drop in the CPU gap partly offset by 17 ms more wait.
4. **The PR #24 stack on this hardware**: 503-token TTFT 188.5 → 72.2 s
   (2.6×) via layer-major chunk 32 (GPU execution 75.8 → 40.4 s, 20 623 →
   656 command buffers); token-major to token-major 188.5 → 163.8 s (−13%)
   with identical GPU time (75.97 vs 75.82 s), from 51 → 41 command buffers
   per token and a 78.6 → 52.8 s CPU gap; decode +12% long context (2.24 →
   2.52) and +4–8% short context (2.91/2.93 → 3.13/3.09 in the fast
   page-cache state, 2.79 → 2.91 in the faulting state), with 41 instead of
   51 buffers per step and identical greedy output.
5. The M4 doc deferred S4 (async expert-miss loading) pending "a trace from
   a RAM-constrained target (… `--cache-gb 2` … on a small-RAM Mac where the
   pool cannot live in page cache) [that] shows the cold-decode profile as
   steady state". This is that trace: page cache 6–10 GiB against a
   16.9 GiB pool, 9–19 GiB of SSD reads per 503-token run at steady state,
   and a CPU gap of 148–253 ms per decode step that tracks the cache budget.

## S3c follow-up: the CPU gap sub-attributed and cut (`4c6fd79` → `83ce946`)

Run 2026-09-02 16:04 → 16:30 EDT on the same box, model, prompts, flags,
sampler, and 1-warmup + 3-measured discipline as above (`--cache-gb 4`,
64 new tokens, greedy, fresh process per run under `/usr/bin/time -l`,
soft idle gate, pre/post `vm_stat`). Three builds, all from this branch,
built before the first timed run:

- **before** `4c6fd79`: the `1c158d8` stack above plus the S3c
  instrumentation only (no cuts). Its decode rates reproduce the `5d8b2ab`
  cells (short 2.90 / 3.10 / 3.11 vs 2.91 / 3.13 / 3.09; long 2.52 / 2.51 /
  2.52 vs 2.52 / 2.54 / 2.47), so the instrumentation costs nothing visible.
- **after** `83ce946`: before + the three cuts below.
- **serial** (A/B arm, not committed): after with the concurrent preads
  forced serial, to separate the fetch cut's two parts. Warmup + 2 measured.

### What the gap contained (before, per decode step = 64-step totals ÷ 64)

S3c (`StepMetrics.cpuGap`) times the CPU work the step performs between
buffers with the same clock as encode and wait; the scopes are disjoint, so
their sum is a lower bound on wall − wait − encode and the remainder is
"other". The CLI's own argmax and detokenize are timed beside it because
the bench's wall (1 / tok/s) contains them too.

| Component (before, warm-1) | 16-tok prompt | 503-tok prompt |
|---|---|---|
| Step wall | 320.6 ms | 394.1 ms |
| Blocking wait | 196.8 ms | 202.2 ms |
| Encode (buffer creation → commit) | 7.6 ms | 6.7 ms |
| **CPU gap (wall − wait − encode)** | **116.2 ms (36%)** | **185.2 ms (47%)** |
| — expert-cache fetch | 89.3 ms (52 misses / step) | 160.2 ms (139 misses / step) |
| — embedding row lookup | 21.1 ms | 18.4 ms |
| — router readback + softmax + top-k | 0.7 ms | 0.6 ms |
| — KV mirror append | 0.3 ms | 1.5 ms |
| — command buffer + encoder creation | 1.3 ms | 1.3 ms |
| — `commit()` | 1.1 ms | 1.0 ms |
| — logits readback | 0.1 ms | 0.1 ms |
| — other (unattributed) | 2.3 ms | 2.1 ms |
| CLI argmax (outside the model) | 1.7 ms | 1.7 ms |
| CLI detokenize + print | 0.6 ms | 0.4 ms |

Two items were the gap: the fetch (77–87%, 1.2–1.7 ms per miss — one
pread at a time against the SSD, plus a victim scan over two dictionaries
for all 2427 slots on every miss) and the embedding lookup (10–18%), which
turned out to copy the entire quantized `embed_tokens` tensor (254 MB +
2 × 16 MB of scales/biases) through `Data.subdata` to extract one row, on
every token — 5.0 s of the 503-token prefill as well. The three cuts:

1. `ebf25f0` — safetensors sub-range reads in place (memcpy / unaligned
   loads over the mapped file; the data section is not word-aligned).
2. `0f39424` — the cache resolves a batch's hits and victims first, then
   issues the misses as concurrent preads; the victim scan reads two flat
   per-slot arrays. Same policy, decision for decision (tested against a
   transcription of the old one over 400 evicting requests).
3. `83ce946` — the KV mirror appends in place instead of copying the whole
   history each step.

Nothing about the schedule changed: 41 command buffers and 2082 dispatches
per decode step, 656 / 148 809 per 503-token prefill, same as `5d8b2ab`.

### A — 16-token prompt, `--cache-gb 4`, layer-major chunk 32

| Build / run | TTFT | Decode tok/s | Decode wait / GPU (64 steps) | Gap (64 steps) | of which fetch / embedding / KV mirror | Hits / misses | SSD pageins | Peak footprint / RSS | user / sys | Real |
|---|---|---|---|---|---|---|---|---|---|---|
| before warm-0 (warmup) | 3.492 s | **2.91** | 13.870 / 9.953 s | 7.528 s | 5.878 / 1.301 / 0.020 s | 17152 / 5437 (76%) | 5.9 GiB | 5.47 / 6.81 GiB | 5.2 / 7.7 s | 27.34 s |
| before warm-1 | 3.260 s | **3.10** | 12.597 / 9.746 s | 7.435 s | 5.716 / 1.351 / 0.021 s | 17152 / 5437 (76%) | 4.7 GiB | 5.47 / 6.82 GiB | 5.4 / 6.2 s | 25.29 s |
| before warm-2 | 3.093 s | **2.90** | 14.060 / 9.917 s | 7.404 s | 5.778 / 1.286 / 0.020 s | 17152 / 5437 (76%) | 4.8 GiB | 5.47 / 6.41 GiB | 5.1 / 7.7 s | 27.50 s |
| before warm-3 | 3.230 s | **3.11** | 12.508 / 9.697 s | 7.455 s | 5.750 / 1.340 / 0.022 s | 17152 / 5437 (76%) | 4.7 GiB | 5.47 / 6.82 GiB | 5.4 / 6.2 s | 25.19 s |
| after warm-0 (warmup) | 2.388 s | **3.65** | 12.557 / 8.944 s | 4.191 s | 3.756 / 0.007 / 0.009 s | 17152 / 5437 (76%) | 6.2 GiB | 5.22 / 6.29 GiB | 3.2 / 8.3 s | 22.73 s |
| after warm-1 | 2.160 s | **3.87** | 11.803 / 8.815 s | 3.968 s | 3.546 / 0.007 / 0.007 s | 17152 / 5437 (76%) | 3.6 GiB | 5.22 / 6.30 GiB | 3.1 / 8.0 s | 20.02 s |
| after warm-2 | 2.084 s | **3.65** | 12.654 / 9.098 s | 4.129 s | 3.691 / 0.007 / 0.007 s | 17152 / 5437 (76%) | 3.6 GiB | 5.22 / 6.29 GiB | 3.2 / 9.2 s | 21.81 s |
| after warm-3 | 2.194 s | **3.85** | 11.851 / 8.910 s | 4.021 s | 3.596 / 0.007 / 0.007 s | 17152 / 5437 (76%) | 3.5 GiB | 5.22 / 6.30 GiB | 3.2 / 8.1 s | 20.17 s |
| serial warm-0 (warmup) | 3.162 s | **3.45** | 12.307 / 9.408 s | 5.507 s | 5.100 / 0.007 / 0.007 s | 17152 / 5437 (76%) | 5.6 GiB | 5.23 / 6.31 GiB | 3.0 / 6.9 s | 23.06 s |
| serial warm-1 | 2.728 s | **3.28** | 13.197 / 9.336 s | 5.576 s | 5.153 / 0.006 / 0.008 s | 17152 / 5437 (76%) | 4.3 GiB | 5.23 / 6.30 GiB | 3.0 / 7.8 s | 24.40 s |
| serial warm-2 | 2.888 s | **3.47** | 12.215 / 9.294 s | 5.506 s | 5.097 / 0.007 / 0.007 s | 17152 / 5437 (76%) | 4.2 GiB | 5.22 / 6.30 GiB | 3.0 / 6.9 s | 22.68 s |

Major page faults (`/usr/bin/time -l`): before 63 / 42 034 / 36, after
36 / 36 457 / 37, serial 36 006 / 36 — the same two page-cache states as the
campaign above, so compare like with like: fast state 3.10–3.11 → 3.85–3.87
(+24–25%), faulting state 2.90 → 3.65 (+26%). TTFT 3.09–3.26 → 2.08–2.19 s
(−33%; the prefill's 16 embedding rows and 2109 misses).

### C — 503-token prompt, `--cache-gb 4`, layer-major chunk 32

| Build / run | TTFT | Prefill wait / GPU | Prefill gap (fetch / embedding) | Decode tok/s | Decode wait / GPU (64 steps) | Gap (64 steps) | of which fetch / embedding / KV mirror | Hits / misses | SSD pageins | Peak footprint / RSS | user / sys | Real |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before warm-0 (warmup) | 72.132 s | 46.371 / 40.360 s | 25.550 s (20.314 / 4.952) | **2.50** | 13.623 / 10.348 s | 11.382 s | 9.840 / 1.140 / 0.092 s | 36452 / 27497 (57%) | 15.8 GiB | 5.59 / 6.45 GiB | 17.0 / 23.9 s | 99.88 s |
| before warm-1 | 70.555 s | 45.632 / 40.332 s | 24.708 s (19.461 / 4.962) | **2.52** | 12.941 / 10.317 s | 11.852 s | 10.254 / 1.175 / 0.094 s | 36452 / 27497 (57%) | 14.6 GiB | 5.59 / 6.51 GiB | 17.3 / 22.1 s | 97.29 s |
| before warm-2 | 72.198 s | 46.362 / 40.400 s | 25.626 s (20.391 / 4.950) | **2.51** | 13.585 / 10.283 s | 11.335 s | 9.802 / 1.135 / 0.089 s | 36452 / 27497 (57%) | 16.4 GiB | 5.59 / 6.35 GiB | 16.9 / 23.9 s | 99.80 s |
| before warm-3 | 70.696 s | 45.609 / 40.356 s | 24.874 s (19.622 / 4.969) | **2.52** | 12.905 / 10.304 s | 11.884 s | 10.280 / 1.183 / 0.094 s | 36452 / 27497 (57%) | 14.6 GiB | 5.59 / 6.51 GiB | 17.3 / 22.3 s | 97.42 s |
| after warm-0 (warmup) | 60.860 s | 49.171 / 40.260 s | 11.206 s (10.705 / 0.027) | **3.13** | 12.914 / 9.468 s | 6.680 s | 6.142 / 0.007 / 0.008 s | 36452 / 27497 (57%) | 11.5 GiB | 5.31 / 5.82 GiB | 8.6 / 41.1 s | 83.38 s |
| after warm-1 | 58.109 s | 46.699 / 40.295 s | 10.861 s (10.339 / 0.029) | **3.29** | 12.077 / 9.499 s | 6.540 s | 6.034 / 0.008 / 0.008 s | 36452 / 27497 (57%) | 11.5 GiB | 5.32 / 6.31 GiB | 9.1 / 36.0 s | 78.94 s |
| after warm-2 | 60.932 s | 49.157 / 40.303 s | 11.299 s (10.798 / 0.027) | **3.12** | 13.113 / 9.478 s | 6.602 s | 6.071 / 0.007 / 0.008 s | 36452 / 27497 (57%) | 12.7 GiB | 5.31 / 5.85 GiB | 8.6 / 40.5 s | 83.71 s |
| after warm-3 | 58.025 s | 46.720 / 40.304 s | 10.774 s (10.251 / 0.034) | **3.29** | 12.057 / 9.472 s | 6.543 s | 6.026 / 0.009 / 0.009 s | 36452 / 27497 (57%) | 11.4 GiB | 5.32 / 6.30 GiB | 9.1 / 35.9 s | 78.83 s |
| serial warm-0 (warmup) | 67.380 s | 46.822 / 40.324 s | 20.297 s (19.952 / 0.017) | **2.66** | 13.474 / 9.986 s | 9.928 s | 9.509 / 0.006 / 0.007 s | 36452 / 27497 (57%) | 15.9 GiB | 5.31 / 6.04 GiB | 6.7 / 28.6 s | 94.00 s |
| serial warm-1 | 65.612 s | 44.762 / 40.252 s | 20.541 s (20.143 / 0.019) | **2.74** | 12.501 / 9.998 s | 10.144 s | 9.715 / 0.007 / 0.007 s | 36452 / 27497 (57%) | 14.3 GiB | 5.31 / 6.20 GiB | 7.2 / 26.2 s | 90.31 s |
| serial warm-2 | 67.221 s | 46.698 / 40.292 s | 20.257 s (19.913 / 0.016) | **2.67** | 13.458 / 9.954 s | 9.885 s | 9.463 / 0.006 / 0.007 s | 36452 / 27497 (57%) | 15.9 GiB | 5.30 / 5.64 GiB | 6.6 / 28.5 s | 93.45 s |

Major page faults: before 36 / 38 438 / 26, after 33 / 36 526 / 36, serial
36 / 37 062. Like with like: fast state 2.52 → 3.29 (+31%), faulting state
2.51 → 3.12 (+24%). TTFT 70.6–72.2 → 58.0–60.9 s (−16 to −18%).

### Decode step after (per step, warm-1)

| Component (after) | 16-tok prompt | 503-tok prompt |
|---|---|---|
| Step wall | 255.6 ms (was 320.6) | 301.4 ms (was 394.1) |
| Blocking wait | 184.4 ms | 188.7 ms |
| GPU (sum of buffer durations) | 137.7 ms (was 152.3) | 148.4 ms (was 161.2) |
| Encode | 9.2 ms (was 7.6) | 10.5 ms (was 6.7) |
| **CPU gap** | **62.0 ms (24%; was 116.2)** | **102.2 ms (34%; was 185.2)** |
| — fetch | 55.4 ms (was 89.3) | 94.3 ms (was 160.2) |
| — embedding | 0.1 ms (was 21.1) | 0.1 ms (was 18.4) |
| — KV mirror | 0.1 ms (was 0.3) | 0.1 ms (was 1.5) |
| — router / cb setup / commit / logits | 0.8 / 1.5 / 1.3 / 0.1 ms | 0.9 / 1.9 / 1.6 / 0.1 ms |
| — other | 2.7 ms | 3.2 ms |

Reading it:

- **The fetch cut splits cleanly.** Serial arm vs before isolates the flat
  victim scan (+ the embedding and KV cuts, which the serial arm also has):
  fetch 5.72–5.78 → 5.10–5.15 s short (−11%, ~10 ms/step), 9.80–10.28 →
  9.46–9.72 s long (−4%). Concurrent preads then take it to 3.55–3.69 s
  (−30%) and 6.03–6.07 s (−37%). Per miss: 1.72 → 1.08 ms short, 1.15 →
  0.68 ms long. In the prefill, where a layer's union has ~29 misses, the
  concurrent fetch halves the fetch (19.5–20.4 → 10.3–10.8 s) and moves
  its 31 GiB of nominal misses at ~3 GiB/s, which is this SSD — that arm is
  now bandwidth-bound, not latency-bound.
- **The embedding cut is total**: 1.29–1.35 s → 0.007 s per 64 steps,
  4.95–4.97 s → 0.03 s per 503-token prefill. Peak footprint falls 0.25 GiB
  (5.47 → 5.22, 5.59 → 5.32 GiB): the transient 254 MB copy is gone.
- **Three things moved that the cuts do not directly touch, reported not
  explained.** Decode GPU time (sum of buffer durations) fell 9.70–9.92 →
  9.29–9.34 (serial arm) → 8.82–9.10 s (short) and 10.28–10.32 → 9.95–10.00
  → 9.47–9.50 s (long) — the same 2624 buffers, same dispatches; the CPU no
  longer streams 286 MB per token through unified memory next to the GPU,
  and the concurrent build changes the read pattern again, which are the
  obvious candidates, but the counters cannot confirm either. Encode rose 0.44–0.49 → 0.59–0.67 s (+2–4
  ms/step) and "other" 0.13 → 0.17–0.21 s (+0.5–1 ms/step); both are small
  against the gain and both appear with the concurrent-pread build only
  (serial arm: encode 0.52–0.57 s). SSD pageins per run fell 4.7–4.8 →
  3.5–3.6 GiB (short) and 14.6–16.4 → 11.4–12.7 GiB (long) on the
  concurrent build while the serial arm stayed at before's level; user
  CPU time fell (5.4 → 3.2 s, 17 → 9 s) and sys time rose (6.2 → 8.0 s,
  22 → 36–40 s) as the preads moved to pool threads.
- **What is left.** The gap is 24% / 34% of a step and 89–92% of it is the
  fetch, now 0.7–1.1 ms per miss with 1.3 / 3.5 misses per layer — SSD
  reads, serial across layers by the routing dependency. Everything else
  in the gap is under 3 ms/step. The wait (184–189 ms/step, of which GPU
  138–148 ms) is now the largest term; (wait − GPU) is still ~1 ms per
  command buffer.

### Cuts considered and not taken (numbers from the before split)

- **GPU argmax (b)**: the CLI argmax is 1.7 ms/step (0.5% of a step) and
  the logits readback 0.1 ms; a GPU reduction would save at most that
  minus one dispatch, and would need first-max tie semantics pinned for
  greedy parity. Not worth the surface change at this size.
- **Fewer commit+wait points (d)**: creation + `commit()` are 2.4 ms/step
  (0.75%); the per-layer round trip is forced by the CPU top-k → fetch
  dependency, so cutting buffers means moving that dependency, not
  merging waits. Rejected: the S3a/S3b baselines stay pinned unchanged.
- **Skipping the KV mirror memcpy (c)**: the mirror is read by the
  attention parity tests and by S6 state persistence, so it stays; the
  in-place append removes the O(history) part and leaves 0.1 ms/step.
- **Router readback batching (a, second form)**: 0.6–0.7 ms/step; nothing
  to batch.
- **Overlapping the fetch with GPU work (a, first form)**: not possible
  as stated — layer L's routing needs layer L's attention output, and
  every later GPU dispatch depends on layer L's experts. What remains
  overlappable is the ~1 ms command-buffer submission latency against the
  fetch tail (an `MTLSharedEvent` gate in front of each MoE), which would
  be S4 proper; untested here.

### Output parity

Every measured run's stdout is byte-identical to the corresponding
`5d8b2ab` bench log: md5 `6986955b…` for the 16-token prompt (all before
/ after / serial runs, equal to `stack-A-c4-warm-*`) and `0a2e3e68…` for
the 503-token prompt (equal to `stack-C32-c4-warm-*`). Cache hit/miss
counts are identical run for run. Fixture-scale walls (tiny-model-q4, raw
and repacked, 200 greedy tokens, 3 runs each) are within noise — 131–135
→ 135–137 tok/s raw, 163–166 → 165–167 tok/s qpack — with identical
output, as expected for a 128 × 64 embedding and 64 resident blobs.

Raw logs: `~/build/cpugap-swiftlet/.bench/logs/` on macmini2
(`{before,after,serial}-{A,C32}-c4-warm-N.{out,err,pre,post}`,
`campaign-{before,after}.log`, `fixture-walls.log`).
