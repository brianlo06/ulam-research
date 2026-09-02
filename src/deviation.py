import numpy as np, sys, json, lzma
a = np.fromfile(sys.argv[1], dtype=np.uint32).astype(np.int64)
lam = float(np.load("data/lambda12.npy")[0])
N = len(a); TWO_PI = 2*np.pi
g = np.diff(a)
out = {}

# ---- deviation process e_n = a_n - n/d  (d = empirical density) ----
d = N/a[-1]
e = a - np.arange(1, N+1)/d
print(f"density={d:.8f}   e_n: min={e.min():.1f} max={e.max():.1f} std={e.std():.2f}")
# growth of running max |e| with n
ns = np.unique(np.logspace(2, np.log10(N-1), 40).astype(int))
grow = [(int(n), float(np.abs(e[:n]).max())) for n in ns]
out['dev_growth'] = grow
for n, m in grow[::8]: print(f"  max|e| over first {n:>8d} terms: {m:8.1f}")

# ---- spectrum of e_n (is there a second hidden frequency?) ----
w = e - e.mean()
W = np.abs(np.fft.rfft(w*np.hanning(N)))**2
fr = np.fft.rfftfreq(N)  # cycles per index n
top = np.argsort(-W[1:])[:12]+1
print("\ntop spectral peaks of e_n (freq in cycles/term, period in terms):")
for k in top:
    print(f"  f={fr[k]:.6f}  period={1/fr[k]:10.1f} terms  power={W[k]:.3e}")
out['e_spectrum_top'] = [(float(fr[k]), float(W[k])) for k in top]

# lam/d mod 2pi: the mean phase drift per term
drift = (lam/d) % TWO_PI
print(f"\nmean theta drift per term lam/d mod 2pi = {drift:.6f} rad ({drift/TWO_PI:.6f} cycles)")
out['drift_cycles'] = float(drift/TWO_PI)

# ---- entropy rate ladder on gap symbols ----
gv, gi = np.unique(g, return_inverse=True); NG=len(gv); M=len(g)
def H_cond_idx(x, y, ny, idx):
    xv, xc = np.unique(x[idx], return_inverse=True)
    joint = np.zeros((len(xv), ny)); np.add.at(joint,(xc,y[idx]),1)
    pj = joint/joint.sum(); px = pj.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.nansum(pj*np.log2(pj/px[:,None]))
def extrap(x, y, ny):
    f = H_cond_idx(x,y,ny,slice(None))
    h = 0.5*(H_cond_idx(x,y,ny,slice(0,M//2))+H_cond_idx(x,y,ny,slice(M//2,None)))
    return 2*f-h, f
print("\nentropy-rate ladder H(g | k previous gaps):")
ctx = np.zeros(M, dtype=np.int64); ladder=[]
for k in range(0,6):
    if k>0:
        ctx = ctx*(NG+1)
        pk = np.full(M,0,dtype=np.int64); pk[k:] = gi[:-k]+1
        ctx = ctx + pk
    est, plug = extrap(ctx, gi, NG)
    ladder.append((k, float(est), float(plug)))
    print(f"  k={k}: plug-in={plug:.4f}  extrapolated={est:.4f}")
out['ladder'] = ladder

# LZMA compression upper bound on entropy rate
comp = lzma.compress(gi.astype(np.uint8).tobytes(), preset=9)
rate = 8*len(comp)/M
print(f"LZMA bound: {rate:.4f} bits/gap")
out['lzma_rate'] = float(rate)

# ---- gap alphabet growth ----
alpha = [(int(n), int(len(np.unique(g[:n])))) for n in ns]
out['alphabet_growth'] = alpha
print("\ngap alphabet size vs n:", [f"{n}:{s}" for n,s in alpha[::8]])
json.dump(out, open("data/deviation12.json","w"), indent=1)
