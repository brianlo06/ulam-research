# Anatomy of a Hidden Signal

An experimental-mathematics dissection of the **Ulam sequence** — 1, 2, 3, 4, 6, 8, 11, 13, 16, 18, 26, …
(each term the smallest integer expressible as a sum of two distinct earlier terms in *exactly one way*).

In 2015, Steinerberger discovered that a mysterious constant λ ≈ 2.5714474995 makes the residues
λ·aₙ mod 2π cluster on a fixed arc of the circle — a "hidden signal" in a sequence long thought
patternless, and still unexplained. This project asks a question the literature hadn't:
**how much of the sequence's behaviour does the signal actually govern?**

📄 **Paper (interactive figures):** [`paper/index.html`](paper/index.html) — self-contained HTML, open locally.

## Results

Computed from the 2,959,301 Ulam numbers below 4×10⁷, with every element's unique representation
recorded at generation time, plus an 80-family scan of alternative starting pairs U(u,v):

| # | Finding |
|---|---------|
| R1 | **Replication at 10-digit agreement** — independent pipeline reproduces the Gibbs–McCranie wavelength 2.443442967…, the exception set {2, 3, 47, 69}, and the pure harmonic comb (extended here with a per-gap-class spectral test). |
| R2 | **The signal carries half the local information** — the circle coordinate θ explains 1.61 bits = 48.2% of next-gap entropy, and screens off **93.5%** of the dependence between consecutive gaps: θ is an approximate sufficient statistic for short-range dynamics. |
| R3 | **…but the rest looks like genuine diffusion** — ~1 bit/gap of apparent entropy survives depth-6 context and LZMA; block-sum variance grows ∝ m⁰·⁹⁵ (Hurst H ≈ 0.48). The Ulam sequence is *not* a bounded-remainder set / quasicrystal: it is a rigid one-frequency skeleton occupied stochastically. |
| R4 | **The generative oligarchy** — a circle-geometry lemma (verified exactly on all 2.96M elements) shows the middle-third bulk cannot reproduce itself; growth is administered by 340 "outlier" elements (~n⁰·³⁰), whose top **ten** members fill 48.5% of all representation slots. The four classical exceptions {2, 3, 47, 69} are exactly the four dominant generators. |
| R5 | **The one-third law** — across all 63 signal-carrying families, window occupancy is **0.3312 ± 0.0007** of the circle, while λ, density, and entropy vary freely. One-third is precisely the extremal measure at which a window W can satisfy (W+W) ∩ W = ∅ mod 1: Ulam sequences saturate the self-exclusion bound. |

Five falsifiable conjectures are stated in §7 of the paper.

## Layout

```
src/ulam.c        fast generator: incremental bitset convolution, cost Σ min(aₙ, L−aₙ)/64 words
src/ulam2.c       same + records each element's unique representation pair (b, aₙ−b)
src/*.py          analysis: λ refinement, entropy/MI with split-half bias control, spectroscopy,
                  fluctuation scaling, representation census, family battery, paper build
tests/            brute-force verifier used by CI
data/             derived JSON results (large .bin term files are regenerable, not committed)
paper/            the paper — index.html is self-contained with all figure data embedded
```

## Reproduce

```sh
cc -O3 -march=native -o src/ulam  src/ulam.c
cc -O3 -march=native -o src/ulam2 src/ulam2.c

src/ulam2 1 2 40000000 data/u12r_4e7.bin        # ~9 min: terms + representations to 4×10⁷
src/ulam  1 2 40000000 data/u12_4e7.bin         # (or copy the ulam2 output minus .rep)
src/ulam  1 2 10000000 data/u12_1e7.bin
python3 src/refine_lambda.py data/u12_1e7.bin   # λ from scratch
python3 src/confirm4e7.py                       # headline numbers + shuffle null (seed 0)
# family scan (~40 min): see src/family_scan.sh + src/scan2.sh, then src/family_analyze.py
python3 src/gen_paper_data.py && python3 src/build_paper.py
```

Whole pipeline: ~1 hour on an 8-core laptop, deterministic throughout. Requires only a C compiler
and Python 3 + numpy (analysis); the CI verifier is stdlib-only.

## References

Steinerberger, *A Hidden Signal in the Ulam Sequence*, Exp. Math. 26 (2017) — [arXiv:1507.00267](https://arxiv.org/abs/1507.00267) ·
Gibbs & McCranie, *The Ulam Numbers up to One Trillion* — [viXra:1711.0134](https://vixra.org/abs/1711.0134) ·
Knuth, [ULAM-GIBBS](https://www-cs-faculty.stanford.edu/~knuth/programs/ulam-gibbs.w) ·
Kravitz & Steinerberger — [arXiv:1705.01883](https://arxiv.org/abs/1705.01883) ·
Hinman, Kuca, Schlesinger & Sheydvasser — [arXiv:1711.00145](https://arxiv.org/abs/1711.00145)

MIT © 2026 Brian Lo · analysis and paper produced with [Claude Code](https://claude.com/claude-code)
