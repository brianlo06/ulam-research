"""Brute-force check of ulam2 output: sequence correctness + unique-representation pairs.

Stdlib only, so it runs anywhere. Recomputes the Ulam sequence naively and checks
(1) the generated terms match, (2) each recorded parent pair is the element's
unique representation as a sum of two distinct earlier terms.
"""
import struct, sys

path = sys.argv[1]
raw = open(path, "rb").read()
terms = list(struct.unpack(f"<{len(raw)//4}I", raw))
reps = list(struct.unpack(f"<{len(open(path+'.rep','rb').read())//4}I",
                          open(path + ".rep", "rb").read()))
assert len(terms) == len(reps)

limit = terms[-1]
# naive reference generator
seq = [terms[0], terms[1]]
S = set(seq)
for m in range(terms[1] + 1, limit + 1):
    count = sum(1 for x in seq if x < m - x and (m - x) in S)
    if count == 1:
        seq.append(m); S.add(m)
assert seq == terms, f"sequence mismatch: first diff at index {next(i for i,(a,b) in enumerate(zip(seq,terms)) if a!=b)}"

for i in range(2, len(terms)):
    m, b = terms[i], reps[i]
    c = m - b
    assert b in S and c in S and b < c, f"bad rep for {m}: ({b},{c})"
    pairs = [(x, m - x) for x in seq if x < m - x and (m - x) in S]
    assert pairs == [(b, c)], f"rep for {m} not unique/mismatched: {pairs} vs ({b},{c})"

print(f"OK: {len(terms)} terms verified, all {len(terms)-2} representation pairs unique and correct")
