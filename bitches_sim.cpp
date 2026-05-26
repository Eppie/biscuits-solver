// bitches (a dice game) — Monte Carlo simulator
// Glue Bunny Games. Rules (verbatim, via Larry Waldman / Dave Fetterman's paper):
//
//   "12 regular dice + 1 8 sided die, 1 10 sided die, 1 12 sided die. Roll all
//    the dice. You must take at least one die out each roll. If the die (or
//    dice) you remove are not their max value (eg 6 for a regular die), you get
//    max - <die value> amount of points. ... Points are bad. After all the dice
//    are gone, add up your points and that is your score."
//
// Model: 15 dice (12*d6, d8, d10, d12). penalty(die) = maxFace - shownFace.
// Score = sum of penalties. Lower is better.
//
// Strategy ("take exactly 1 die"): each roll remove exactly one die => exactly
// 15 rolls/game. Greedy choice = the die with the smallest current penalty
// (closest to its own max). Ties break toward the LARGEST die (most dangerous
// to keep) by ordering big dice first and keeping the first-seen minimum.
//
// Performance:
//  - Games are SIMD lanes; the innermost loop is over games and is purely
//    elementwise -> auto-vectorizes (NEON / AVX).
//  - Each game's SURVIVING dice are kept compacted to the front of a per-game
//    size array, so roll r only scans (15-r) slots instead of all 15. The
//    alive-count is identical across all games each roll, so the loop bound
//    stays uniform across lanes. Total inner iterations drop 225 -> 120/game,
//    and this removes the alive-mask, bit-extract, and dead-die select. The
//    only per-lane scalar work is moving the last live die into the freed slot.
//  - Per-lane PRNG is 32-bit xorshift (4-wide on NEON), seeded by splitmix64.
//  - Die faces via Lemire multiply-shift (no modulo); and since
//    penalty = max - face is itself uniform on [0, max-1], we sample penalty
//    directly: pen = (rand32 * size) >> 32.
//  - Cache-resident tiling. Single-threaded (parallelism intentionally left off).
//
// Build (x86):  c++ -O3 -march=native -funroll-loops -fopenmp-simd -o bitches_sim bitches_sim.cpp
// Build (ARM):  c++ -O3 -mcpu=native  -funroll-loops -fopenmp-simd -o bitches_sim bitches_sim.cpp
// Run:          ./bitches_sim [num_games] [seed]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>

static constexpr int NUM_DICE = 15;
// Big dice first so strict-< min keeps the largest die on a penalty tie.
alignas(64) static constexpr int32_t MAXV[NUM_DICE] = {
    12, 10, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
};
static constexpr int   TILE = 2048;     // games per cache-resident tile (fits L1)
static constexpr int32_t BIG = 1 << 20; // "infinite" penalty sentinel

// splitmix64: used only to expand a seed into PRNG state.
static inline uint64_t splitmix64(uint64_t &x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// Cheap 32-bit xorshift (Marsaglia). 3 shifts + 3 xors, elementwise -> 4-wide
// on NEON. Quality is plenty for estimating a dice-score mean; streams are
// decorrelated by splitmix64 seeding. State must be non-zero.
static inline uint32_t xorshift32(uint32_t &x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// THR[s][D] = max penalty at which to BANK a die of size s when D dice remain.
// From single-die optimal stopping: bank iff penalty <= W[s][D-1], where
//   W[s][1] = (s-1)/2,  W[s][m] = mean_p( min(p, W[s][m-1]) ),
// i.e. accept now if it beats the value of continuing optimally. D==1 forces a
// bank (it's the last roll for that die). Horizon D = dice-left is an upper
// bound on a kept die's remaining rolls (the Fetterman "parallel games" model).
static int THR[13][NUM_DICE + 1];
static void init_thr() {
    for (int s : {6, 8, 10, 12}) {
        double W[NUM_DICE + 1];
        W[1] = (s - 1) / 2.0;
        for (int m = 2; m <= NUM_DICE; ++m) {
            double cont = W[m - 1], acc = 0;
            for (int p = 0; p < s; ++p) acc += (p < cont ? p : cont);
            W[m] = acc / s;
        }
        for (int D = 0; D <= NUM_DICE; ++D)
            THR[s][D] = (D <= 1) ? s : (int)W[D - 1];  // floor(W); D==1 => always bank
    }
}

struct Stats {
    uint64_t sum = 0, sumsq = 0;
    int32_t  smin = INT32_MAX, smax = 0;
    uint64_t hist[128] = {};
    void merge(const Stats &o) {
        sum += o.sum; sumsq += o.sumsq;
        if (o.smin < smin) smin = o.smin;
        if (o.smax > smax) smax = o.smax;
        for (int i = 0; i < 128; ++i) hist[i] += o.hist[i];
    }
};

// Simulate games [lo, hi) into `st`.
static void run_range(uint64_t lo, uint64_t hi, uint64_t base_seed, Stats &st) {
    alignas(64) static uint32_t rng[TILE];
    alignas(64) static int32_t  score[TILE], minkey[TILE], argslot[TILE];
    // dsz[d][i]: size of the d-th still-alive die of game i. SoA so the
    // inner game loop is unit-stride. Alive dice are kept compacted to [0,count).
    alignas(64) static uint8_t  dsz[NUM_DICE][TILE];

    for (uint64_t done = lo; done < hi; done += TILE) {
        const int n = (int)((hi - done < (uint64_t)TILE) ? (hi - done) : TILE);

        // --- init tile ---
        for (int i = 0; i < n; ++i) {
            uint64_t seed = base_seed ^ (done + (uint64_t)i + 0x9E3779B97F4A7C15ULL);
            uint32_t s = (uint32_t)splitmix64(seed);
            rng[i]   = s ? s : 0x1234567u;   // xorshift32 needs non-zero state
            score[i] = 0;
        }
        for (int d = 0; d < NUM_DICE; ++d)
            for (int i = 0; i < n; ++i) dsz[d][i] = (uint8_t)MAXV[d];

        // --- 15 rolls; remove exactly one die per roll ---
        for (int count = NUM_DICE; count > 0; --count) {
            #pragma omp simd
            for (int i = 0; i < n; ++i) { minkey[i] = BIG; argslot[i] = 0; }

            for (int d = 0; d < count; ++d) {
                #pragma omp simd
                for (int i = 0; i < n; ++i) {
                    uint32_t r    = xorshift32(rng[i]);
                    int32_t  size = dsz[d][i];
                    // pen = max - face, uniform on [0, size-1] via Lemire map.
                    int32_t  pen  = (int32_t)(((uint64_t)r * (uint64_t)size) >> 32);
                    // Single key: minimize (penalty, -size). pen<<5 dominates
                    // (pen<=11 => step 32 > max size 12), so equal penalties
                    // prefer the LARGEST die. One branchless compare -> vectorizes.
                    int32_t key = (pen << 5) - size;
                    int take   = key < minkey[i];
                    minkey[i]  = take ? key : minkey[i];
                    argslot[i] = take ? d   : argslot[i];
                }
            }

            // Score the removed die, then compact: move the last live die into
            // the freed slot (scalar per-lane move; no-op when it was the last).
            const int last = count - 1;
            for (int i = 0; i < n; ++i) {
                int32_t sz = dsz[argslot[i]][i];          // size of removed die
                score[i]  += (minkey[i] + sz) >> 5;       // recover penalty from key
                dsz[argslot[i]][i] = dsz[last][i];
            }
        }

        // --- reduce tile ---
        for (int i = 0; i < n; ++i) {
            int32_t sc = score[i];
            st.sum   += (uint64_t)sc;
            st.sumsq += (uint64_t)sc * (uint64_t)sc;
            if (sc < st.smin) st.smin = sc;
            if (sc > st.smax) st.smax = sc;
            if (sc < 128) ++st.hist[sc];
        }
    }
}

// Multi-take threshold strategy: each roll, bank EVERY die whose penalty is at
// or below its size/dice-left threshold; if none qualify, bank the single best
// (the "remove >=1" rule). Alive set is a 15-bit mask (variable count banked per
// roll, so games no longer share a uniform alive-count -> mask, not compaction).
static void run_range_threshold(uint64_t lo, uint64_t hi, uint64_t base_seed, Stats &st) {
    alignas(64) static uint32_t rng[TILE], alive[TILE];
    alignas(64) static int32_t  score[TILE], cnt[TILE];
    alignas(64) static int32_t  minpen[TILE], argmin[TILE], bankmask[TILE], banksum[TILE];
    alignas(64) static int32_t  mindiff[TILE];   // min (penalty - threshold) for forced bank
    // Per-roll thresholds by die size, precomputed from cnt[] so the hot slot
    // loop does a plain unit-stride load instead of a cnt-indexed gather.
    alignas(64) static int32_t  thr[13][TILE];
    constexpr uint32_t FULL = (1u << NUM_DICE) - 1;

    for (uint64_t done = lo; done < hi; done += TILE) {
        const int n = (int)((hi - done < (uint64_t)TILE) ? (hi - done) : TILE);

        for (int i = 0; i < n; ++i) {
            uint64_t seed = base_seed ^ (done + (uint64_t)i + 0x9E3779B97F4A7C15ULL);
            uint32_t s = (uint32_t)splitmix64(seed);
            rng[i]   = s ? s : 0x1234567u;
            score[i] = 0; alive[i] = FULL; cnt[i] = NUM_DICE;
        }

        // At most NUM_DICE rolls (>=1 banked each); finished games (alive==0)
        // contribute nothing on later rolls.
        for (int roll = 0; roll < NUM_DICE; ++roll) {
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                int c = cnt[i];
                thr[6][i] = THR[6][c]; thr[8][i] = THR[8][c];
                thr[10][i] = THR[10][c]; thr[12][i] = THR[12][c];
                mindiff[i] = BIG; minpen[i] = 0; argmin[i] = 0; bankmask[i] = 0; banksum[i] = 0;
            }

            for (int d = 0; d < NUM_DICE; ++d) {
                const int32_t size = MAXV[d];
                const int32_t *tt = thr[size];               // plain load, no gather
                #pragma omp simd
                for (int i = 0; i < n; ++i) {
                    uint32_t r   = xorshift32(rng[i]);
                    int32_t  pen = (int32_t)(((uint64_t)r * (uint64_t)size) >> 32);
                    int bit      = (int)((alive[i] >> d) & 1u);
                    int isbank   = bit & (pen <= tt[i]);
                    bankmask[i] |= isbank << d;
                    banksum[i]  += isbank ? pen : 0;
                    // Forced-bank fallback (Fetterman step 5): of the dice that
                    // missed, take the one that "differs from its cutoff by the
                    // least", i.e. minimize (penalty - threshold).
                    int diff = bit ? (pen - tt[i]) : BIG;
                    int take = diff < mindiff[i];
                    mindiff[i] = take ? diff : mindiff[i];
                    minpen[i]  = take ? pen  : minpen[i];
                    argmin[i]  = take ? d    : argmin[i];
                }
            }

            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                int force = (alive[i] != 0) & (bankmask[i] == 0);  // must remove >=1
                bankmask[i] = force ? (1 << argmin[i]) : bankmask[i];
                banksum[i]  = force ? minpen[i]        : banksum[i];
                score[i]   += banksum[i];
                alive[i]   &= ~(uint32_t)bankmask[i];
                cnt[i]     -= __builtin_popcount((uint32_t)bankmask[i]);
            }
        }

        for (int i = 0; i < n; ++i) {
            int32_t sc = score[i];
            st.sum   += (uint64_t)sc;
            st.sumsq += (uint64_t)sc * (uint64_t)sc;
            if (sc < st.smin) st.smin = sc;
            if (sc > st.smax) st.smax = sc;
            if (sc < 128) ++st.hist[sc];
        }
    }
}

int main(int argc, char **argv) {
    uint64_t num_games = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 10'000'000ULL;
    uint64_t base_seed = (argc > 2) ? strtoull(argv[2], nullptr, 10) : 0xC0FFEEULL;
    // mode: "take1" (default) or "threshold" (any arg containing 'h', e.g. "thr").
    bool threshold = (argc > 3) && strpbrk(argv[3], "hH") != nullptr;
    const char *label = threshold
        ? "multi-take threshold (optimal-stopping per die)"
        : "take exactly 1 die/roll (min-penalty greedy)";

    init_thr();
    auto t0 = std::chrono::steady_clock::now();

    Stats g;
    if (threshold) run_range_threshold(0, num_games, base_seed, g);
    else           run_range(0, num_games, base_seed, g);

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    double mean = (double)g.sum / (double)num_games;
    double var  = (double)g.sumsq / (double)num_games - mean * mean;
    double sd   = var > 0 ? sqrt(var) : 0.0;

    auto pctl = [&](double p) -> double {
        uint64_t target = (uint64_t)(p * (double)num_games);
        uint64_t cum = 0;
        for (int s = 0; s < 128; ++s) { cum += g.hist[s]; if (cum >= target) return s; }
        return g.smax;
    };

    printf("bitches (a dice game) — strategy: %s\n", label);
    printf("games      : %llu\n", (unsigned long long)num_games);
    printf("mean score : %.4f\n", mean);
    printf("median     : %.0f\n", pctl(0.5));
    printf("std dev    : %.4f\n", sd);
    printf("quartiles  : p25=%.0f  p50=%.0f  p75=%.0f  p90=%.0f  p99=%.0f\n",
           pctl(0.25), pctl(0.50), pctl(0.75), pctl(0.90), pctl(0.99));
    printf("min / max  : %d / %d\n", g.smin, g.smax);
    printf("time       : %.3f s  (%.1f M games/s)\n", secs, num_games / secs / 1e6);

    printf("\nscore distribution:\n");
    int lo = g.smin, hi = g.smax < 127 ? g.smax : 127;
    uint64_t peak = 1;
    for (int s = lo; s <= hi; ++s) if (g.hist[s] > peak) peak = g.hist[s];
    for (int s = lo; s <= hi; ++s) {
        if (!g.hist[s]) continue;
        int bar = (int)(50.0 * g.hist[s] / peak);
        printf("%3d | %6.3f%% %.*s\n", s, 100.0 * (double)g.hist[s] / (double)num_games,
               bar, "##################################################");
    }
    return 0;
}
