import numpy as np, sys
a = np.fromfile(sys.argv[1], dtype=np.uint32).astype(np.float64)
N = len(a)
print(f"N={N}  a_max={a[-1]:.0f}")

def power(lam):
    # |mean exp(i lam a_n)|
    ph = np.mod(lam*a, 2*np.pi)
    return abs(np.mean(np.cos(ph)) + 1j*np.mean(np.sin(ph)))

lam0 = 2.5714474995
# coarse local scan
for width, npts in [(1e-6, 401), (1e-8, 401), (1e-10, 401)]:
    grid = lam0 + np.linspace(-width, width, npts)
    vals = np.array([power(l) for l in grid])
    lam0 = grid[np.argmax(vals)]
    print(f"  width={width:.0e}  lam={lam0:.13f}  |S|={vals.max():.6f}")

# parabolic polish
h = 1e-11
f1,f2,f3 = power(lam0-h), power(lam0), power(lam0+h)
lam = lam0 + h*(f1-f3)/(2*(f1-2*f2+f3))
print(f"refined lambda = {lam:.12f}   |S| = {power(lam):.6f}")
print(f"2*pi/lambda    = {2*np.pi/lam:.12f}")
np.save("data/lambda12.npy", np.array([lam]))
