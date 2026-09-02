import numpy as np, json, glob, os, sys
TWO_PI = 2*np.pi

def analyze(path):
    a = np.fromfile(path, dtype=np.uint32).astype(np.int64)
    if len(a) < 20000: return None
    L = int(a[-1])+1; N = len(a)
    g = np.diff(a)
    row = dict(file=os.path.basename(path), N=N, L=L, density=N/L)
    # regularity test: eventually periodic gaps? check if the last 30% of gap seq is periodic
    tail = g[int(0.7*len(g)):]
    per = None
    for p in range(1, min(4000, len(tail)//3)):
        if np.all(tail[p:] == tail[:-p]): per = p; break
    row['periodic'] = per is not None
    row['period'] = per
    # signal detection via FFT of indicator
    ind = np.zeros(L); ind[a] = 1.0
    F = np.abs(np.fft.rfft(ind))/N; F[0]=0
    k = int(np.argmax(F))
    lam0 = TWO_PI*k/L
    # refine by local maximization
    def power(l):
        ph = np.mod(l*a, TWO_PI)
        return abs(np.mean(np.cos(ph))+1j*np.mean(np.sin(ph)))
    lam = lam0
    for w in [TWO_PI/L, TWO_PI/L/50]:
        grid = lam + np.linspace(-w, w, 81)
        vals = [power(x) for x in grid]
        lam = grid[int(np.argmax(vals))]
    row['lam'] = float(lam); row['S'] = float(power(lam))
    # take fundamental: check subharmonics lam/2, lam/3 stronger?
    for div in [2,3,4,5]:
        if power(lam/div) > row['S']*0.7:
            l2 = lam/div
            for w in [1e-5, 1e-7]:
                grid = l2 + np.linspace(-w,w,81); vals=[power(x) for x in grid]
                l2 = grid[int(np.argmax(vals))]
            if power(l2) > row['S']*0.7:
                row['lam'] = float(l2); row['S'] = float(power(l2)); break
    if row['S'] < 0.05 or row['periodic']:
        return row  # no meaningful signal / regular
    th = np.mod(row['lam']*a, TWO_PI)[:-1]
    # info quantities (128 theta bins, extrapolated)
    gv, gi = np.unique(g, return_inverse=True); NG=len(gv); M=len(g)
    NB=128
    tb = np.minimum((th/TWO_PI*NB).astype(int), NB-1)
    def Hc(x, idx):
        xv, xc = np.unique(x[idx], return_inverse=True)
        y = gi[idx]
        joint = np.zeros((len(xv), NG)); np.add.at(joint,(xc,y),1)
        pj = joint/joint.sum(); px = pj.sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return -np.nansum(pj*np.log2(pj/px[:,None]))
    def ex(x):
        f = Hc(x, slice(None)); h = 0.5*(Hc(x, slice(0,M//2))+Hc(x, slice(M//2,None)))
        return 2*f-h
    z = np.zeros(M, dtype=np.int64)
    Hg = ex(z)
    Hth = ex(tb)
    prev = np.zeros(M,dtype=np.int64); prev[1:] = gi[:-1]+1
    Hg1 = ex(prev)
    Hthg1 = ex(tb*(NG+2)+prev)
    row.update(H_g=float(Hg), H_g_th=float(Hth), I_th=float(Hg-Hth),
               I_frac=float((Hg-Hth)/Hg) if Hg>0 else None,
               I_g1=float(Hg-Hg1),
               screen=float(1-(Hth-Hthg1)/max(Hg-Hg1,1e-9)))
    # window measure: fraction of circle occupied (99.9% mass interval)
    lo, hi = np.quantile(np.mod(th,TWO_PI), [0.0005, 0.9995])
    row['window_frac'] = float((hi-lo)/TWO_PI)
    # rescaled window profile (universality test): map [q.001,q.999]->[0,1]
    lo9, hi9 = np.quantile(th, [0.001, 0.999])
    u = (th - lo9)/(hi9-lo9)
    hp, _ = np.histogram(u[(u>=0)&(u<=1)], bins=100, range=(0,1), density=True)
    row['profile'] = hp.tolist()
    # diffusion ratio at m=1024
    m0=1024; k2=(M//m0)*m0
    S = g[:k2].reshape(-1,m0).sum(1)
    row['diff_ratio'] = float(S.var()/(m0*g.var()))
    return row

rows=[]
for f in sorted(glob.glob("data/fam/u*_*.bin")):
    try:
        r = analyze(f)
    except Exception as e:
        print(f, "ERR", e); continue
    if r:
        rows.append(r)
        print(f"{r['file']:14s} d={r['density']:.4f} {'PER' if r['periodic'] else '   '} lam={r.get('lam',0):.6f} S={r.get('S',0):.3f}"
              + (f" Hg={r['H_g']:.2f} Ifrac={r['I_frac']:.3f} screen={r['screen']:.3f} win={r['window_frac']:.3f} D={r['diff_ratio']:.3f}" if 'H_g' in r else ""))
json.dump(rows, open("data/family_results.json","w"), indent=1)
print(f"\n{len(rows)} families analyzed")
