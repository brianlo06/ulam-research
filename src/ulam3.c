// ulam3.c -- parallel Ulam generator for overnight large-L runs.
// Same incremental bitset-convolution scheme as ulam2.c, with:
//   * the per-term fold parallelized over OUTPUT words via libdispatch
//     (output word k depends only on S[k-wo] and S[k-wo-1]: no carry chain)
//   * unique-representation recording (rep sized to the full bitset span)
//   * incremental streaming of (term, rep) pairs to disk + progress logging
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <dispatch/dispatch.h>

static uint64_t *S, *one, *many;
static uint32_t *rep;
static long W;
static uint64_t cur_a;

static inline void apply_word(long k, uint64_t bits) {
    if (!bits || k < 0 || k >= W) return;
    uint64_t m = many[k], o = one[k];
    many[k] = m | (o & bits);
    one[k]  = (o | bits) & ~many[k];
    if (rep) {
        uint64_t fresh = bits & ~o & ~m;
        while (fresh) {
            int t = __builtin_ctzll(fresh);
            uint64_t p = ((uint64_t)k << 6) + t;
            rep[p] = (uint32_t)(p - cur_a);
            fresh &= fresh - 1;
        }
    }
}

// fold output words [k0,k1] (inclusive): rolling single-load of S per word
static void fold_range(long k0, long k1, long wo, long jhi, unsigned sh) {
    long j0 = k0 - wo;
    uint64_t prev = (j0 - 1 >= 0 && j0 - 1 <= jhi) ? S[j0 - 1] : 0;
    for (long k = k0; k <= k1; k++) {
        long j = k - wo;
        uint64_t sw = (j >= 0 && j <= jhi) ? S[j] : 0;
        uint64_t bits = sh ? ((sw << sh) | (prev >> (64 - sh))) : sw;
        apply_word(k, bits);
        prev = sw;
    }
}

static void add_term(uint64_t a, uint64_t L) {
    cur_a = a;
    long wo = (long)(a >> 6);
    unsigned sh = (unsigned)(a & 63);
    uint64_t bhi = (a - 1 < L - a) ? (a - 1) : (L - a);
    long jhi = (long)(bhi >> 6);
    if (jhi >= W) jhi = W - 1;
    long klo = wo, khi = jhi + wo + 1;            // inclusive output word range
    if (khi >= W) khi = W - 1;
    long nk = khi - klo + 1;
    if (nk < 131072) {                             // serial path for small folds (<1 MB span)
        fold_range(klo, khi, wo, jhi, sh);
        return;
    }
    long nstripes = nk / 65536; if (nstripes > 10) nstripes = 10;
    long chunk = (nk + nstripes - 1) / nstripes;
    dispatch_apply((size_t)nstripes, DISPATCH_APPLY_AUTO, ^(size_t s2) {
        long k0 = klo + (long)s2 * chunk;
        long k1 = k0 + chunk - 1; if (k1 > khi) k1 = khi;
        fold_range(k0, k1, wo, jhi, sh);
    });
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ulam3 u v L [out.bin]\n"); return 1; }
    uint64_t u = strtoull(argv[1], 0, 10);
    uint64_t v = strtoull(argv[2], 0, 10);
    uint64_t L = strtoull(argv[3], 0, 10);
    const char *out = (argc > 4) ? argv[4] : NULL;

    W = (long)(L / 64) + 4;
    S    = calloc(W, 8);
    one  = calloc(W, 8);
    many = calloc(W, 8);
    int norep = (argc > 5 && strcmp(argv[5], "norep") == 0);
    rep = norep ? NULL : calloc((size_t)W * 64, 4);
    if (!S || !one || !many || (!rep && !norep)) { fprintf(stderr, "oom\n"); return 1; }

    FILE *ft = NULL, *fr = NULL;
    char rp[512];
    if (out) {
        ft = fopen(out, "wb");
        if (rep) { snprintf(rp, sizeof rp, "%s.rep", out); fr = fopen(rp, "wb"); }
        if (!ft || (rep && !fr)) { fprintf(stderr, "cannot open output\n"); return 1; }
    }
    #define EMIT(t) do { \
        if (ft) { uint32_t _t = (uint32_t)(t); fwrite(&_t, 4, 1, ft); \
                  if (fr) { uint32_t _r = rep[(t)]; fwrite(&_r, 4, 1, fr); } } } while (0)

    long n = 0;
    time_t t0 = time(0);
    add_term(u, L); S[u >> 6] |= 1ULL << (u & 63); EMIT(u); n++;
    add_term(v, L); S[v >> 6] |= 1ULL << (v & 63); EMIT(v); n++;

    for (uint64_t m = v + 1; m <= L; m++) {
        long k = (long)(m >> 6); unsigned b = (unsigned)(m & 63);
        if (((one[k] >> b) & 1ULL) == 0) continue;
        add_term(m, L);
        S[k] |= 1ULL << b;
        EMIT(m); n++;
        if ((n & 0xFFFFF) == 0) {                  // progress every ~1M terms
            fflush(ft); fflush(fr);
            fprintf(stderr, "[%ld s] n=%ld  m=%llu (%.1f%%)\n",
                    (long)(time(0) - t0), n, (unsigned long long)m, 100.0 * m / L);
            fflush(stderr);
        }
    }
    if (ft) fclose(ft);
    if (fr) fclose(fr);
    fprintf(stderr, "DONE U(%llu,%llu) L=%llu : N=%ld density=%.8f  elapsed=%ld s\n",
            (unsigned long long)u, (unsigned long long)v, (unsigned long long)L,
            n, (double)n / (double)L, (long)(time(0) - t0));
    return 0;
}
