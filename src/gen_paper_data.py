import numpy as np, json, os
TWO_PI = 2*np.pi
lam = float(np.load("data/lambda12.npy")[0])
a = np.fromfile("data/u12_4e7.bin", dtype=np.uint32).astype(np.int64)
N = len(a)
rho = np.mod(lam*a, TWO_PI)/TWO_PI
g = np.diff(a)
D = {'lam': lam, 'wavelength': TWO_PI/lam, 'N': N, 'amax': int(a[-1]), 'density': N/a[-1]}

# hero scatter
D['hero'] = json.load(open("data/hero.json"))

# window profile at 4e7
h, e = np.histogram(rho, bins=200, range=(0,1), density=True)
D['profile'] = {'x': np.round(e[:-1]+0.0025, 4).tolist(), 'y': np.round(h,3).tolist()}

# gap distribution (top 16 + other)
gv, gc = np.unique(g, return_counts=True)
o = np.argsort(-gc)
top = o[:16]
D['gapdist'] = {'g': gv[top].tolist(), 'p': np.round(gc[top]/gc.sum(),5).tolist(),
                'other': float(1-gc[top].sum()/gc.sum())}
# wrapped step per top gap
wr = (lam*gv[top]) % TWO_PI; wr = np.where(wr>np.pi, wr-TWO_PI, wr)
D['gapdist']['wrapped'] = np.round(wr/TWO_PI,4).tolist()   # in cycles

# p(g | rho) heatmap: 96 rho bins over [0.30,0.70], top 10 gaps ordered by wrapped step
NB=96; lo,hi=0.30,0.70
sel = (rho[:-1]>=lo)&(rho[:-1]<hi)
tb = ((rho[:-1][sel]-lo)/(hi-lo)*NB).astype(int).clip(0,NB-1)
top10 = gv[o[:10]]
order = np.argsort((lam*top10 % TWO_PI + np.pi) % TWO_PI)  # order rows by wrapped step
top10 = top10[order]
gg = g[sel]
heat = np.zeros((len(top10), NB))
tot = np.zeros(NB); np.add.at(tot, tb, 1)
for i,g0 in enumerate(top10):
    np.add.at(heat[i], tb[gg==g0], 1)
heat = heat/np.maximum(tot,1)
D['heat'] = {'gaps': top10.tolist(), 'lo': lo, 'hi': hi, 'nb': NB,
             'p': np.round(heat,3).tolist(),
             'occ': np.round(tot/tot.max(),3).tolist()}

# info numbers
C = json.load(open("data/confirm4e7.json"))
D['info'] = C

# diffusion curve: combine 1e7 fine + 4e7 coarse
vg = g.var(); diff=[]
for m in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]:
    k=(len(g)//m)*m; S=g[:k].reshape(-1,m).sum(1)
    diff.append((m, float(S.var()/(m*vg)), len(S)))
D['diffusion'] = diff

# deviation trace
d = N/a[-1]
ev = a - np.arange(1,N+1)/d
idx = np.linspace(0, N-1, 3000).astype(int)
D['dev'] = {'n': idx.tolist(), 'e': np.round(ev[idx],1).tolist()}

# oligarchy: from rep 4e7 if available else 1e7
repf = "data/u12r_4e7.bin" if os.path.exists("data/u12r_4e7.bin.rep") else "data/u12r_1e7.bin"
ar = np.fromfile(repf, dtype=np.uint32).astype(np.int64)
rr = np.fromfile(repf+".rep", dtype=np.uint32).astype(np.int64)
D['rep_source_N'] = len(ar)
rhor = np.mod(lam*ar, TWO_PI)/TWO_PI
inl = (rhor>=1/3)&(rhor<=2/3)
b = rr[2:]; m2 = ar[2:]
allsum = np.concatenate([b, m2-b])
uniq, deg = np.unique(allsum, return_counts=True)
deg_of = np.zeros(len(ar), dtype=np.int64); deg_of[np.searchsorted(ar, uniq)] = deg
dd = np.sort(deg_of[deg_of>0])[::-1]
csum = np.cumsum(dd)/(2*len(b))
Kpts = np.unique(np.logspace(0, np.log10(len(dd)-1), 120).astype(int))
D['share_curve'] = {'K': Kpts.tolist(), 'share': np.round(csum[Kpts-1],4).tolist()}
D['zipf'] = {'rank': Kpts.tolist(), 'deg': dd[Kpts-1].tolist()}
# outlier birth
oi = np.where(~inl)[0]
ns = np.unique(np.logspace(1, np.log10(len(ar)-1), 60).astype(int))
D['birth'] = {'n': ns.tolist(), 'count': [int((~inl[:n]).sum()) for n in ns]}
# top generators table
topg = np.argsort(-deg_of)[:10]
D['topgen'] = [(int(ar[i]), float(np.round(rhor[i],4)), int(deg_of[i]), float(np.round(deg_of[i]/(2*len(b)),4))) for i in topg]
# rep caste stats at this scale
rho_of = np.zeros(int(ar[-1])+1); rho_of[ar] = rhor
in_b = (rho_of[b]>=1/3)&(rho_of[b]<=2/3); in_c = (rho_of[m2-b]>=1/3)&(rho_of[m2-b]<=2/3)
D['repstats'] = {
  'one_outlier': float(np.mean(in_b ^ in_c)), 'both_out': float(np.mean(~in_b & ~in_c)),
  'both_in': int((in_b & in_c).sum()), 'n_outliers': int((~inl).sum()),
  'outlier_smaller': float(np.mean(~in_b & in_c)/max(np.mean((~in_b & in_c))+np.mean((in_b & ~in_c)),1e-12)),
  'sterile': float((deg_of==0).mean())}
# b vs gap identity check
bv, bc = np.unique(b, return_counts=True)
gv2, gc2 = np.unique(np.diff(ar), return_counts=True)
gm = dict(zip(gv2.tolist(), gc2.tolist()))
D['b_vs_gap'] = [(int(x), int(c), int(gm.get(int(x),0))) for x,c in sorted(zip(bv,bc), key=lambda t:-t[1])[:8]]
# alphabet growth
D['alphabet'] = {'n': ns.tolist(), 'size': [int(len(np.unique(np.diff(ar[:n])))) for n in ns]}

# families
if os.path.exists("data/family_results.json"):
    fam = json.load(open("data/family_results.json"))
    for r in fam: r.pop('profile_full', None)
    D['families'] = fam
json.dump(D, open("data/paper_data.json","w"))
print("wrote paper_data.json", os.path.getsize("data/paper_data.json"), "bytes")
