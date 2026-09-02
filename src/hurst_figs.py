import numpy as np, sys, json
a = np.fromfile(sys.argv[1], dtype=np.uint32).astype(np.int64)
lam = float(np.load("data/lambda12.npy")[0])
N=len(a); TWO_PI=2*np.pi
g = np.diff(a); th = np.mod(lam*a, TWO_PI)
out={}

# ---- fluctuation scaling of gap partial sums (DFA-0 / aggregated variance) ----
# S_m = sum of m consecutive gaps; compare Var(S_m) to iid prediction m*Var(g)
vg = g.var(); print(f"Var(g)={vg:.2f} mean(g)={g.mean():.4f}")
rows=[]
for m in [1,2,4,8,16,32,64,128,256,512,1024,4096,16384]:
    k = (len(g)//m)*m
    S = g[:k].reshape(-1,m).sum(1)
    ratio = S.var()/(m*vg)
    rows.append((m, float(S.var()), float(ratio)))
    print(f"  m={m:6d}  Var(S_m)={S.var():12.1f}  Var/(m*Var(g))={ratio:.4f}")
out['fluct'] = rows

# effective Hurst from largest scales
ms = np.array([r[0] for r in rows[4:]]); vs = np.array([r[1] for r in rows[4:]])
slope = np.polyfit(np.log(ms), np.log(vs), 1)[0]
print(f"log-log slope of Var(S_m) at large m: {slope:.3f}  (2H={slope:.3f} -> H={slope/2:.3f})")
out['hurst'] = float(slope/2)

# ---- figure data ----
# theta histogram (the hidden-signal window density)
hist, edges = np.histogram(th, bins=240, range=(0,TWO_PI), density=True)
out['theta_hist'] = {'x': edges[:-1].tolist(), 'y': hist.tolist()}

# p(g|theta): heat map over 120 theta bins x top 12 gaps
gv, cnt = np.unique(g, return_counts=True)
topg = gv[np.argsort(-cnt)[:12]]
NBH=120
tb = np.minimum((th[:-1]/TWO_PI*NBH).astype(int), NBH-1)
heat = np.zeros((NBH, len(topg)))
other = np.zeros(NBH); tot = np.zeros(NBH)
np.add.at(tot, tb, 1)
for j,g0 in enumerate(topg):
    np.add.at(heat[:,j], tb[g==g0], 1)
heat_n = heat/np.maximum(tot,1)[:,None]
out['heat'] = {'gaps': topg.tolist(), 'theta_centers': ((np.arange(NBH)+0.5)*TWO_PI/NBH).tolist(),
               'p': heat_n.tolist(), 'occupancy': (tot/tot.sum()).tolist()}

# top-1 gap per bin and its conditional probability (deterministic skeleton)
best = heat_n.argmax(1); bestp = heat_n.max(1)
occ = tot > tot.sum()/NBH*0.1
print(f"\ndeterministic skeleton: over occupied bins, mean top-1 p(g|theta)={bestp[occ].mean():.3f}")
out['skeleton'] = {'best_gap': [int(topg[b]) for b in best], 'best_p': bestp.tolist(), 'occupied': occ.tolist()}

# gap distribution
out['gap_dist'] = {'g': gv.tolist(), 'p': (cnt/cnt.sum()).tolist()}
# wrapped step for each gap
wr = (lam*gv) % TWO_PI; wr[wr>np.pi] -= TWO_PI
out['gap_wrapped'] = wr.tolist()

# deviation trace (subsampled) for figure
d = N/a[-1]
e = a - np.arange(1,N+1)/d
idx = np.linspace(0, N-1, 4000).astype(int)
out['dev_trace'] = {'n': idx.tolist(), 'e': e[idx].tolist()}

# entropy ladder + info numbers assembled elsewhere
json.dump(out, open("data/figs12.json","w"))
print("wrote data/figs12.json")
