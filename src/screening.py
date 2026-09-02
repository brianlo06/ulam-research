import numpy as np, sys, json
a = np.fromfile(sys.argv[1], dtype=np.uint32).astype(np.int64)
lam = float(np.load("data/lambda12.npy")[0])
TWO_PI = 2*np.pi
th = np.mod(lam*a, TWO_PI)[:-1]
g  = np.diff(a)
N = len(g)
gv, gi = np.unique(g, return_inverse=True)
NG = len(gv)
NB = 128   # theta bins - modest to control bias
tb = np.minimum((th/TWO_PI*NB).astype(int), NB-1)

def H_cond(xi, nx, y_i, ny, idx=None):
    """plug-in H(y|x) in bits over sample subset idx"""
    if idx is None: idx = slice(None)
    x = xi[idx]; y = y_i[idx]
    joint = np.zeros((nx, ny))
    np.add.at(joint, (x, y), 1)
    n = joint.sum()
    pj = joint/n
    px = pj.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.nansum(pj*np.log2(pj/px[:,None]))

def extrap(fn):
    """jackknife-style bias control: value at N, N/2, N/4 -> Richardson extrapolate.
    plug-in bias ~ -c/n so H_true ~ 2*H(N) - H(N/2) using two halves averaged"""
    full = fn(slice(None))
    halves = [fn(slice(0, N//2)), fn(slice(N//2, None))]
    half = np.mean(halves)
    return 2*full - half, full, half

# quantities, all as conditional entropies of g_n
combos = {}
z = np.zeros(N, dtype=int)
combos['H(g)']                    = (z, 1)
combos['H(g|theta)']              = (tb, NB)
prev = np.full(N, -1); prev[1:] = gi[:-1]
c1 = (prev+1); combos['H(g|g1)'] = (c1, NG+1)
prev2 = np.full(N, -1); prev2[2:] = gi[:-2]
c2 = (prev+1)*(NG+1)+(prev2+1); combos['H(g|g1,g2)'] = (c2, (NG+1)**2)
c3 = tb*(NG+1)+(prev+1); combos['H(g|theta,g1)'] = (c3, NB*(NG+1))
c4 = (tb*(NG+1)+(prev+1))*(NG+1)+(prev2+1); combos['H(g|theta,g1,g2)'] = (c4, NB*(NG+1)**2)
prev3 = np.full(N, -1); prev3[3:] = gi[:-3]
c5 = c2*(NG+1)+(prev3+1); combos['H(g|g1,g2,g3)'] = (c5, (NG+1)**3)

out = {}
for name, (xi, nx) in combos.items():
    # compact the index space
    xv, xc = np.unique(xi, return_inverse=True)
    est, full, half = extrap(lambda idx, xc=xc, nx=len(xv): H_cond(xc, nx, gi, NG, idx))
    out[name] = dict(extrapolated=float(est), plugin=float(full), half=float(half))
    print(f"{name:22s}  plug-in={full:.4f}  extrapolated={est:.4f}  (bias~{full-est:+.4f})")

Hg = out['H(g)']['extrapolated']
Hth = out['H(g|theta)']['extrapolated']
Hg1 = out['H(g|g1)']['extrapolated']
Hg12 = out['H(g|g1,g2)']['extrapolated']
Hthg1 = out['H(g|theta,g1)']['extrapolated']
print()
print(f"I(g;theta)        = {Hg-Hth:.4f} bits ({100*(Hg-Hth)/Hg:.1f}% of H(g))")
print(f"I(g;g1)           = {Hg-Hg1:.4f} bits")
print(f"I(g;g1|theta)     = {Hth-Hthg1:.4f} bits  -> screening ratio {(1-(Hth-Hthg1)/(Hg-Hg1))*100:.1f}%")

# long-range dependence: I(g_n; g_{n-k}) raw vs conditioned on theta
print("\nlag-k gap dependence (extrapolated):")
lr = []
for k in [1,2,3,5,8,13,21,34,55]:
    pk = np.full(N,-1); pk[k:] = gi[:-k]
    xv, xc = np.unique(pk+1, return_inverse=True)
    e1,_,_ = extrap(lambda idx, xc=xc: H_cond(xc, len(xv), gi, NG, idx))
    ck = tb*(NG+1)+(pk+1)
    xv2, xc2 = np.unique(ck, return_inverse=True)
    e2,_,_ = extrap(lambda idx, xc2=xc2: H_cond(xc2, len(xv2), gi, NG, idx))
    I_raw = Hg - e1; I_cond = Hth - e2
    lr.append((k, I_raw, I_cond))
    print(f"  k={k:3d}: I(g;g_-k)={I_raw:.4f}   I(g;g_-k|theta)={max(I_cond,0):.4f}")
out['longrange'] = lr
json.dump(out, open("data/screening12.json","w"), indent=1)
