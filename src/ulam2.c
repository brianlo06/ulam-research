// ulam2.c -- fast generator for Ulam-type sequences U(u,v)
// U(u,v): a1=u, a2=v; a_{n+1} = least integer > a_n expressible as a sum of two
// DISTINCT earlier terms in exactly one way.
//
// Method: incremental bitset convolution. We keep three bitsets over [0,L]:
//   S    : membership of the sequence
//   one  : integers with exactly 1 representation so far
//   many : integers with >= 2 representations so far
// When a term a joins S, every pair (b,a) with b in S, b<a is realised for the
// first time, so we OR (S << a) into the representation counters. All such sums
// exceed a, hence lie strictly ahead of the scan frontier: the scheme is exact.
//
// Cost is sum_n min(a_n, L-a_n)/64 words, not N*L/64.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint64_t *S, *one, *many;
static uint32_t *rep;   // rep[p] = smaller summand of p's first representation
static long W;

static uint64_t cur_a;   // term being added (larger summand)
static inline void apply_word(long k, uint64_t bits) {
    if (!bits || k < 0 || k >= W) return;
    uint64_t m = many[k], o = one[k];
    uint64_t fresh = bits & ~o & ~m;      // positions gaining their FIRST rep
    many[k] = m | (o & bits);
    one[k]  = (o | bits) & ~many[k];
    while (fresh) {                        // record smaller summand b = p - a
        int t = __builtin_ctzll(fresh);
        uint64_t p = ((uint64_t)k << 6) + t;
        rep[p] = (uint32_t)(p - cur_a);
        fresh &= fresh - 1;
    }
}

// fold (S << a) into the counters
static void add_term(uint64_t a, uint64_t L) {
    cur_a = a;
    long wo = (long)(a >> 6);
    unsigned sh = (unsigned)(a & 63);
    uint64_t bhi = (a - 1 < L - a) ? (a - 1) : (L - a);   // largest useful b
    long jhi = (long)(bhi >> 6);
    if (jhi >= W) jhi = W - 1;
    uint64_t carry = 0;
    for (long j = 0; j <= jhi; j++) {
        uint64_t sw = S[j];
        uint64_t lo = (sw << sh) | carry;
        carry = sh ? (sw >> (64 - sh)) : 0;
        apply_word(j + wo, lo);
    }
    apply_word(jhi + 1 + wo, carry);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ulam u v L [out.bin]\n"); return 1; }
    uint64_t u = strtoull(argv[1], 0, 10);
    uint64_t v = strtoull(argv[2], 0, 10);
    uint64_t L = strtoull(argv[3], 0, 10);
    const char *out = (argc > 4) ? argv[4] : NULL;

    W = (long)(L / 64) + 4;
    S    = calloc(W, 8);
    one  = calloc(W, 8);
    many = calloc(W, 8);
    rep  = calloc((size_t)W * 64, 4);   /* bits can land in padding words past L */
    if (!S || !one || !many || !rep) { fprintf(stderr, "oom\n"); return 1; }

    long cap = 1 << 20, n = 0;
    uint64_t *terms = malloc(cap * sizeof(uint64_t));

    // seed u, then v (adding v realises the pair (u,v))
    add_term(u, L); S[u >> 6] |= 1ULL << (u & 63); terms[n++] = u;
    add_term(v, L); S[v >> 6] |= 1ULL << (v & 63); terms[n++] = v;

    for (uint64_t m = v + 1; m <= L; m++) {
        long k = (long)(m >> 6); unsigned b = (unsigned)(m & 63);
        if (((one[k] >> b) & 1ULL) == 0) continue;      // 0 or >=2 reps
        add_term(m, L);
        S[k] |= 1ULL << b;
        if (n == cap) { cap <<= 1; terms = realloc(terms, cap * sizeof(uint64_t)); }
        terms[n++] = m;
    }

    if (out) {
        FILE *f = fopen(out, "wb");
        for (long i = 0; i < n; i++) { uint32_t t = (uint32_t)terms[i]; fwrite(&t, 4, 1, f); }
        fclose(f);
        char rp[512]; snprintf(rp, sizeof rp, "%s.rep", out);
        f = fopen(rp, "wb");
        for (long i = 0; i < n; i++) { uint32_t r = rep[terms[i]]; fwrite(&r, 4, 1, f); }
        fclose(f);
    }
    fprintf(stderr, "U(%llu,%llu) up to %llu : N=%ld  density=%.8f  last=%llu\n",
            (unsigned long long)u, (unsigned long long)v, (unsigned long long)L,
            n, (double)n / (double)L, (unsigned long long)terms[n-1]);
    // first 20 terms for sanity
    fprintf(stderr, "head:");
    for (long i = 0; i < (n < 20 ? n : 20); i++) fprintf(stderr, " %llu", (unsigned long long)terms[i]);
    fprintf(stderr, "\n");
    return 0;
}
