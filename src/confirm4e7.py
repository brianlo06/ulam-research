import numpy as np, json
a = np.fromfile("data/u12_4e7.bin", dtype=np.uint32).astype(np.int64)
N=len(a); TWO_PI=2*np.pi
out={'N':N}

# refine lambda on the big set
def power(l, arr):
    ph = np.mod(l*arr, TWO_PI); return abs(np.mean(np.cos(ph))+1j*np.mean(np.sin(ph)))
lam = 2.571447498685
for w,npts in [(3e-9,41),(3e-10,41),(3e-11,41)]:
    grid = lam+np.linspace(-w,w,npts)
    vals=[power(x,a) for x in grid]; lam=grid[int(np.argmax(vals))]
print(f"lambda(4e7) = {lam:.13f}   |S|={power(lam,a):.6f}")
print(f"wavelength  = {TWO_PI/lam:.13f}  (Gibbs: 2.443442967784743)")
out['lambda']=float(lam); out['S']=float(power(lam,a)); out['wavelength']=float(TWO_PI/lam)
np.save("data/lambda12.npy", np.array([lam]))

th = np.mod(lam*a, TWO_PI); rho = th/TWO_PI
exc = a[np.cos(th)>=0]
print(f"cos>=0 exceptions among {N} terms: {exc.tolist()[:10]} (count={len(exc)})")
out['exceptions']=exc.tolist()[:10]; out['n_exc']=int(len(exc))
inl = (rho>=1/3)&(rho<=2/3)
print(f"outliers(strict middle-third def): {(~inl).sum()}")
out['outliers_4e7'] = int((~inl).sum())

g = np.diff(a)
gv = np.unique(g)
print(f"gap alphabet: {len(gv)}  max gap: {g.max()}")
out['alphabet']=int(len(gv)); out['max_gap']=int(g.max())

# diffusion scaling to larger m
vg=g.var(); rows=[]
for m in [256,1024,4096,16384,65536]:
    k=(len(g)//m)*m; S=g[:k].reshape(-1,m).sum(1)
    rows.append((m, float(S.var()/(m*vg))))
    print(f"  m={m:6d}: Var(S_m)/(m Var g) = {S.var()/(m*vg):.4f}   [n_blocks={len(S)}]")
out['diffusion']=rows

# info numbers, extrapolated, 128 bins
gvv, gi = np.unique(g, return_inverse=True); NG=len(gvv); M=len(g)
NB=128
tb = np.minimum((th[:-1]/TWO_PI*NB).astype(int), NB-1)
def Hc(x, idx):
    xv,xc=np.unique(x[idx],return_inverse=True); y=gi[idx]
    j=np.zeros((len(xv),NG)); np.add.at(j,(xc,y),1)
    pj=j/j.sum(); px=pj.sum(1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return -np.nansum(pj*np.log2(pj/px[:,None]))
def ex(x):
    f=Hc(x,slice(None)); h=0.5*(Hc(x,slice(0,M//2))+Hc(x,slice(M//2,None))); return 2*f-h
z=np.zeros(M,dtype=np.int64)
Hg=ex(z); Hth=ex(tb)
prev=np.zeros(M,dtype=np.int64); prev[1:]=gi[:-1]+1
Hg1=ex(prev); Hthg1=ex(tb*(NG+2)+prev)
prev2=np.zeros(M,dtype=np.int64); prev2[2:]=gi[:-2]+1
Hg12=ex(prev*(NG+2)+prev2)
print(f"\nH(g)={Hg:.4f}  H(g|th)={Hth:.4f}  I={Hg-Hth:.4f} ({100*(Hg-Hth)/Hg:.1f}%)")
print(f"I(g;g1)={Hg-Hg1:.4f}  I(g;g1|th)={Hth-Hthg1:.4f}  screening={100*(1-(Hth-Hthg1)/(Hg-Hg1)):.1f}%")
print(f"H(g|g1,g2)={Hg12:.4f}")
out.update(H_g=float(Hg),H_g_th=float(Hth),I_th=float(Hg-Hth),I_g1=float(Hg-Hg1),
           I_g1_th=float(Hth-Hthg1),screen=float(1-(Hth-Hthg1)/(Hg-Hg1)),H_g12=float(Hg12))

# ladder to k=6 on big data
ctx=np.zeros(M,dtype=np.int64); lad=[]
for k in range(0,7):
    if k>0:
        pk=np.zeros(M,dtype=np.int64); pk[k:]=gi[:-k]+1
        ctx=ctx*(NG+1)+pk
        xv,ctx=np.unique(ctx,return_inverse=True)  # compact to avoid overflow
        ctx=ctx.astype(np.int64)
    e=ex(ctx); lad.append((k,float(e)))
    print(f"  ladder k={k}: H(g|k prev)={e:.4f}")
out['ladder']=lad
import lzma
comp=lzma.compress(gi.astype(np.uint16).tobytes(),preset=9)
# uint16: 2 bytes/symbol; compare to raw entropy
rate=8*len(comp)/M
print(f"LZMA(uint16) rate={rate:.4f} bits/gap")
out['lzma']=float(rate)

# ---- NULL CONTROL: shuffle gaps, rebuild, recompute ----
rng=np.random.default_rng(0)
gs = rng.permutation(g)
a2 = np.concatenate([[a[0]], a[0]+np.cumsum(gs)])
th2 = np.mod(lam*a2, TWO_PI)
S2 = power(lam, a2)
tb2 = np.minimum((th2[:-1]/TWO_PI*NB).astype(int), NB-1)
gi2 = np.searchsorted(gvv, gs)
def Hc2(x,y,idx):
    xv,xc=np.unique(x[idx],return_inverse=True); yy=y[idx]
    j=np.zeros((len(xv),NG)); np.add.at(j,(xc,yy),1)
    pj=j/j.sum(); px=pj.sum(1)
    with np.errstate(divide='ignore',invalid='ignore'):
        return -np.nansum(pj*np.log2(pj/px[:,None]))
Hth2 = 2*Hc2(tb2,gi2,slice(None)) - 0.5*(Hc2(tb2,gi2,slice(0,M//2))+Hc2(tb2,gi2,slice(M//2,None)))
print(f"\nNULL (shuffled gaps): |S|={S2:.5f}   I(g;theta)={Hg-Hth2:.5f} bits")
out['null_S']=float(S2); out['null_I']=float(Hg-Hth2)
json.dump(out, open("data/confirm4e7.json","w"), indent=1)
