import numpy as np, sys, json
a = np.fromfile(sys.argv[1], dtype=np.uint32).astype(np.int64)
lam = float(np.load("data/lambda12.npy")[0])
N = len(a); TWO_PI = 2*np.pi
th = np.mod(lam*a, TWO_PI)          # theta_n
g  = np.diff(a)                     # gaps
res = {}

# ---- 1. window / residue distribution ----
frac_neg = np.mean(np.cos(th) < 0)
res['frac_cos_neg'] = float(frac_neg)
exceptions = a[np.cos(th) >= 0]
res['exceptions'] = exceptions[:20].tolist()
res['n_exceptions'] = int(len(exceptions))
print(f"cos(lam a)<0 fraction: {frac_neg:.8f};  exceptions ({len(exceptions)}): {exceptions[:10]}")
print(f"theta range: [{th.min():.4f}, {th.max():.4f}]  (pi/2={np.pi/2:.4f}, 3pi/2={3*np.pi/2:.4f})")

# ---- 2. gap distribution ----
vals, cnts = np.unique(g, return_counts=True)
p = cnts/cnts.sum()
order = np.argsort(-cnts)
print("\nTop gaps (gap, prob, lam*g mod 2pi mapped to (-pi,pi]):")
for i in order[:15]:
    r = (lam*vals[i]) % TWO_PI
    if r > np.pi: r -= TWO_PI
    print(f"  g={vals[i]:4d}  p={p[i]:.5f}  lam*g wrapped={r:+.4f}")
res['gap_support_size'] = int(len(vals))
res['gap_max'] = int(vals.max())
H_g = -np.sum(p*np.log2(p))
res['H_gap_bits'] = float(H_g)
print(f"\ngap support size={len(vals)}, max gap={vals.max()}, H(g)={H_g:.4f} bits")

# ---- 3. mutual information I(theta_n ; g_n) ----
def cond_entropy(x_bins, y, nx):
    # H(y | x binned) using plug-in estimator
    yv, yi = np.unique(y, return_inverse=True)
    ny = len(yv)
    joint = np.zeros((nx, ny))
    np.add.at(joint, (x_bins, yi), 1)
    pj = joint/joint.sum()
    px = pj.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.nansum(pj*np.log2(pj/px[:,None]))
    return h

nb = 256
tb = np.minimum((th[:-1]/TWO_PI*nb).astype(int), nb-1)
H_g_th = cond_entropy(tb, g, nb)
print(f"H(g_n | theta_n) [{nb} bins] = {H_g_th:.4f} bits   -> I = {H_g-H_g_th:.4f} bits")
res['H_g_given_theta'] = float(H_g_th); res['I_theta_gap'] = float(H_g-H_g_th)

# vs. history-based predictors
gv, gi = np.unique(g, return_inverse=True)
H_g_prev = cond_entropy(gi[:-1], g[1:], len(gv))
print(f"H(g_n | g_(n-1))              = {H_g_prev:.4f} bits   -> I = {H_g-H_g_prev:.4f} bits")
res['H_g_given_prevgap'] = float(H_g_prev)
# pair of previous gaps
pair = gi[:-2]*len(gv)+gi[1:-1]
pv, pi_ = np.unique(pair, return_inverse=True)
H_g_prev2 = cond_entropy(pi_, g[2:], len(pv))
print(f"H(g_n | g_(n-1),g_(n-2))      = {H_g_prev2:.4f} bits")
res['H_g_given_prev2'] = float(H_g_prev2)
# theta + prev gap
combo = tb[1:]*len(gv)+gi[:-1]
cv, ci = np.unique(combo, return_inverse=True)
H_g_combo = cond_entropy(ci, g[1:], len(cv))
print(f"H(g_n | theta_n, g_(n-1))     = {H_g_combo:.4f} bits")
res['H_g_given_theta_prevgap'] = float(H_g_combo)

# ---- 4. determinism test: best-guess accuracy ----
def bestguess(x_bins, y, nx):
    yv, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((nx, len(yv)))
    np.add.at(joint, (x_bins, yi), 1)
    return joint.max(1).sum()/len(y)
acc_th = bestguess(tb, g, nb)
acc_prev = bestguess(gi[:-1], g[1:], len(gv))
acc_marg = p.max()
print(f"\nbest-guess accuracy: marginal={acc_marg:.4f}  theta({nb} bins)={acc_th:.4f}  prev gap={acc_prev:.4f}")
res['acc_marginal']=float(acc_marg); res['acc_theta']=float(acc_th); res['acc_prev']=float(acc_prev)

# finer theta bins to see saturation
for nb2 in [16, 64, 256, 1024, 4096]:
    tb2 = np.minimum((th[:-1]/TWO_PI*nb2).astype(int), nb2-1)
    h2 = cond_entropy(tb2, g, nb2)
    ac2 = bestguess(tb2, g, nb2)
    print(f"  bins={nb2:5d}: H(g|theta)={h2:.4f}  I={H_g-h2:.4f}  acc={ac2:.4f}")

json.dump(res, open("data/anatomy12.json","w"), indent=1)
