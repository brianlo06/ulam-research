# Overnight run — 2026-09-03

`caffeinate -i ./src/ulam3 1 2 400000000 data/u12_4e8.bin norep` is generating
U(1,2) up to 4×10⁸ (≈29.6M terms), streaming terms to disk as it goes.
Progress: `tail data/overnight.log`. It survives this Claude session ending;
kill anytime with `pkill ulam3` — the file prefix stays valid.

- Goal: test **Conjecture 3** (diffusive occupancy) two octaves beyond the paper.
- The machine was heavily loaded (Docker VM) at launch; ETA anywhere from ~7h
  (quiet) to well over a day (contended). Quitting Docker Desktop speeds it up a lot.
- Keep the Mac **plugged in** (caffeinate prevents idle sleep, not battery lid-sleep).

## In the morning (works even if the run is still going)
```sh
python3 src/overnight_analysis.py data/u12_4e8.bin
```
Prints λ, exceptions, outlier census, gap stats, the extended diffusion curve
with the effective Hurst exponent, and one-third-law quantiles; writes
`data/overnight_results.json`. Verdict guide: ratio flat near 0.3 at large m
supports "stochastic quasicrystal" (Conj. 3); ratio collapsing toward 0 like
1/m would mean bounded remainder and a major revision of §4.4.
