import numpy as np, sys, json
a = np.fromfile(sys.argv[1], dtype=np.uint32).astype(np.int64)
lam = float(np.load("data/lambda12.npy")[0])
N = len(a); L = int(a[-1])+1
g = np.diff(a)
out = {}

# full-set spectrum via FFT of indicator (replicate: peaks only at k*lam)
ind = np.zeros(L, dtype=np.float64); ind[a] = 1.0
F = np.fft.rfft(ind)
P = np.abs(F)/N
fr = 2*np.pi*np.arange(len(F))/L   # angular frequency
P[0] = 0
top = np.argsort(-P)[:24]
print("full-set spectrum, top peaks (freq, freq/lam, |S|):")
seen = []
for k in sorted(top, key=lambda k:-P[k]):
    f = fr[k]
    if any(abs(f-s)<1e-3 for s in seen): continue
    seen.append(f)
    print(f"  freq={f:.7f}  freq/lam={f/lam:.5f}  |S|={P[k]:.4f}")
out['full_peaks'] = [(float(fr[k]), float(P[k])) for k in top]

# gap-class spectroscopy: indicator of {a_n : gap after a_n = g0}
print("\nper-gap-class spectra (top non-DC peaks):")
gv, cnt = np.unique(g, return_counts=True)
classes = gv[np.argsort(-cnt)[:8]]
cls_out = {}
for g0 in classes:
    pos = a[:-1][g == g0]
    ind2 = np.zeros(L); ind2[pos] = 1.0
    F2 = np.abs(np.fft.rfft(ind2))/len(pos); F2[0]=0
    tk = np.argsort(-F2)[:40]
    peaks=[]; seen=[]
    for k in sorted(tk, key=lambda k:-F2[k]):
        f = fr[k]
        if any(abs(f-s)<2e-3 for s in seen): continue
        seen.append(f); peaks.append((f, F2[k]))
        if len(peaks)>=5: break
    cls_out[int(g0)] = [(float(f), float(v), float(f/lam)) for f,v in peaks]
    ps = "  ".join(f"({f:.5f},{v:.3f},x{f/lam:.3f})" for f,v in peaks)
    print(f"  g={g0:3d} (n={len(pos)}): {ps}")
out['class_peaks'] = cls_out
json.dump(out, open("data/spectroscopy12.json","w"), indent=1)
