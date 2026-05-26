// Monte-Carlo validation of the PROVABLY OPTIMAL policy.
// Compute optimal V (DP), then play games choosing, each roll, the kept-set K
// that maximizes (sum of kept penalties) - V(K). The average score must
// converge to V(12,1,1,1) = 8.0879 if both the DP value and the greedy-on-V
// action are correct.
//
// This variant is the "8-wide over big-die configs" experiment: the three big
// dice (d8,d10,d12) give 8 keep-combinations, and V is laid out so those 8
// values are contiguous, letting one roll run 8 independent max-chains over the
// d6 keep-count (elementwise -> vectorizable) before combining them.
//
// Build: c++ -O3 -mcpu=native -o opt_simd8 opt_simd8.cpp

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>

#include "bitches.h"

static double V[NUM_STATES];    // V[state] = optimal expected remaining score

// xorshift128+ PRNG (Vigna), seeded with fixed constants below.
static uint64_t rngState0 = 0x243F6A8885A308D3ULL, rngState1 = 0x13198A2E03707344ULL;
static inline uint64_t xorshift() {
    uint64_t x = rngState0, y = rngState1;
    rngState0 = y;
    x ^= x << 23;
    rngState1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rngState1 + y;
}
// Returns a penalty uniform on 0..sides-1 (Lemire multiply-shift, no modulo).
static inline int rollDie(int sides) {
    return (int)(((xorshift() >> 32) * (uint64_t)sides) >> 32);
}

int main(int argc, char **argv) {
    long N = (argc > 1) ? atol(argv[1]) : 20000000;
    solveV(V);
    printf("optimal V(12,1,1,1) = %.6f  (target)\n", V[stateIndex(12, 1, 1, 1)]);

    // float copy of V, laid out so the 8 big-die keep-combinations are contiguous.
    static float Vf[104];
    for (int i = 0; i < 104; ++i) Vf[i] = (float)V[i];

    double sum = 0, sumsq = 0;
    long hist[64] = {0};
    double rolls = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (long t = 0; t < N; ++t) {
        int d6 = 12, d8 = 1, d10 = 1, d12 = 1;
        int score = 0;

        while (d6 + d8 + d10 + d12 > 0) {
            rolls++;

            // roll d6 -> histogram of penalties (0..5), build top-k prefix sums (no sort)
            int faceHist[6] = {0, 0, 0, 0, 0, 0};
            for (int j = 0; j < d6; ++j) faceHist[rollDie(6)]++;
            int topPenSum6[13];
            topPenSum6[0] = 0;
            {
                int kept = 0;
                for (int penalty = 5; penalty >= 0; --penalty)
                    for (int c = 0; c < faceHist[penalty]; ++c) {
                        topPenSum6[kept + 1] = topPenSum6[kept] + penalty;
                        ++kept;
                    }
            }
            int pen8  = d8  ? rollDie(8)  : 0;
            int pen10 = d10 ? rollDie(10) : 0;
            int pen12 = d12 ? rollDie(12) : 0;
            int totalPen = topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

            // 8-WIDE over big-configs: V[k6*8 + b] for b=0..7 is contiguous, so
            // run 8 INDEPENDENT max-chains over k6 (elementwise, vectorizable),
            // then combine. bigKeptPen/valid precomputed per roll.
            float bigKeptPen[8];
            int valid[8];
            for (int b = 0; b < 8; ++b) {
                int keep8 = (b >> 2) & 1, keep10 = (b >> 1) & 1, keep12 = b & 1;
                valid[b] = (keep8 <= d8) & (keep10 <= d10) & (keep12 <= d12);
                bigKeptPen[b] = (float)((keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0));
            }
            int bigFullIdx = d8 * 4 + d10 * 2 + d12;
            float maxVal[8];
            int maxKeep6[8];
            for (int b = 0; b < 8; ++b) {
                maxVal[b]   = -1e30f;
                maxKeep6[b] = 0;
            }
            for (int k6 = 0; k6 <= d6; ++k6) {
                float base = (float)topPenSum6[k6];
                const float *vr = Vf + k6 * 8;
                int isFull = (k6 == d6);
                for (int b = 0; b < 8; ++b) {                  // vectorizable, 8 indep chains
                    float v = base + bigKeptPen[b] - vr[b];
                    int ok = valid[b] & ~(isFull & (b == bigFullIdx));
                    v = ok ? v : -1e30f;
                    int better = v > maxVal[b];
                    maxVal[b]   = better ? v  : maxVal[b];
                    maxKeep6[b] = better ? k6 : maxKeep6[b];
                }
            }
            // combine the 8 chains into the single best kept-set
            float best = -1e30f;
            int bestKeep6 = 0, bestBigIdx = 0;
            for (int b = 0; b < 8; ++b) {
                int better = maxVal[b] > best;
                best       = better ? maxVal[b]   : best;
                bestKeep6  = better ? maxKeep6[b] : bestKeep6;
                bestBigIdx = better ? b           : bestBigIdx;
            }
            int keep8 = bestBigIdx >> 2, keep10 = (bestBigIdx >> 1) & 1, keep12 = bestBigIdx & 1;
            int keptPen = topPenSum6[bestKeep6] + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
            score += totalPen - keptPen;                       // banked penalty
            d6 = bestKeep6;
            d8 = keep8;
            d10 = keep10;
            d12 = keep12;
        }
        sum += score;
        sumsq += (double)score * score;
        if (score < 64) hist[score]++;
    }

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mean = sum / N, sd = sqrt(sumsq / N - mean * mean);
    printf("MC loop (excl. DP solve): %.2f s  ->  %.3f M games/s,  %.1f M moves/s  (%.2f rolls/game)\n",
           secs, N / secs / 1e6, rolls / secs / 1e6, rolls / (double)N);
    printf("MC optimal-policy mean = %.5f  (sd %.4f, %ld games, stderr %.5f)\n",
           mean, sd, N, sd / sqrt((double)N));
    printf("difference from exact optimum = %+.5f\n", mean - V[stateIndex(12, 1, 1, 1)]);
    return 0;
}
