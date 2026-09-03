"""Morning-after analysis for the large U(1,2) run (works on a PARTIAL/streaming file).

Usage: python3 src/overnight_analysis.py data/u12_4e8.bin
Targets Conjecture 3 (diffusive occupancy) plus λ refinement, exceptions,
outlier census, alphabet, max gap, and one-third-law window quantiles.
"""
import numpy as np, json, sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "data/u12_4e8.bin"
a = np.fromfile(path, dtype=np.uint32).astype(np.int64)
# streaming file may end mid-write; drop a trailing zero/partial record
while len(a) and (a[-1] == 0 or (len(a) > 1 and a[-1] <= a[-2])): a = a[:-1]
N = len(a); TWO_PI = 2*np.pi
print(f"prefix loaded: N={N:,} terms, a_max={a[-1]:,}  ({a[-1]/4e8*100:.1f}% of target L)")
out = {'N': int(N), 'amax': int(a[-1]), 'density': N/a[-1]}

# lambda refinement (seed from paper value)
def power(l):
    ph = np.mod(l*a, TWO_PI); return abs(np.mean(np.cos(ph))+1j*np.mean(np.sin(ph)))
lam = 2.5714474984285
for w in (3e-10, 3e-11, 3e-12):
    g = lam + np.linspace(-w, w, 41)
    v = [power(x) for x in g]; lam = g[int(np.argmax(v))]
h = 1e-12
f1,f2,f3 = power(lam-h), power(lam), power(lam+h)
lam += h*(f1-f3)/(2*(f1-2*f2+f3))
print(f"lambda = {lam:.13f}   wavelength = {TWO_PI/lam:.13f}   |S|={power(lam):.6f}")
print(f"  (Gibbs-McCranie wavelength: 2.443442967784743)")
out['lambda'] = lam; out['S'] = float(power(lam))

th = np.mod(lam*a, TWO_PI); rho = th/TWO_PI
exc = a[np.cos(th) >= 0]
print(f"exceptions cos>=0: {exc.tolist()[:10]} (count={len(exc)})")
out['exceptions'] = exc.tolist()[:10]
inl = (rho >= 1/3) & (rho <= 2/3)
print(f"outliers: {(~inl).sum()}  (paper: 223 @ N=740k, 340 @ N=2.96M; conjectured ~n^0.30)")
ns = np.unique(np.logspace(2, np.log10(N-1), 40).astype(int))
birth = [(int(n), int((~inl[:n]).sum())) for n in ns]
out['birth'] = birth
if N > 5e6:
    import math
    n1, c1 = birth[-8]; n2, c2 = birth[-1]
    print(f"birth exponent over last stretch: {math.log(c2/c1)/math.log(n2/n1):.3f}")

g = np.diff(a)
print(f"gap alphabet: {len(np.unique(g))}  max gap: {g.max()} (was 114 / 587 at 4e7)")
out['alphabet'] = int(len(np.unique(g))); out['max_gap'] = int(g.max())

# THE HEADLINE: diffusion curve, two octaves beyond the paper
vg = g.var(); rows = []
print("\nVar(S_m)/(m Var g)  [conjecture 3: flat ~0.3 = diffusive; falling like 1/m = bounded]")
m = 256
while (len(g)//m) >= 30:
    k = (len(g)//m)*m
    Sm = g[:k].reshape(-1, m).sum(1)
    r = Sm.var()/(m*vg)
    rows.append((int(m), float(r), int(len(Sm))))
    print(f"  m={m:8d}  ratio={r:.4f}   blocks={len(Sm)}")
    m *= 2
out['diffusion'] = rows
if len(rows) >= 4:
    import math
    (m1, r1, _), (m2, r2, _) = rows[1], rows[-1]
    slope = 1 + math.log(r2/r1)/math.log(m2/m1)
    print(f"effective Var(S_m) ~ m^{slope:.3f}  ->  H = {slope/2:.3f}")
    out['hurst2'] = slope/2

# one-third law window quantiles
for q in (1e-5, 1e-4, 1e-3):
    lo, hi = np.quantile(rho, [q, 1-q])
    print(f"window q={q:.0e}: [{lo:.5f}, {hi:.5f}]  width={hi-lo:.5f} of circle")
out['window_q1e4'] = [float(x) for x in np.quantile(rho, [1e-4, 1-1e-4])]

json.dump(out, open("data/overnight_results.json", "w"), indent=1)
print("\nwrote data/overnight_results.json")
