// Optimal-policy MC, restructured for SIMD-across-games via STATE BUCKETING.
// All games in a bucket share the same state, so the per-move argmax is uniform
// and vectorizes across games with no wasted lanes. Games flow strictly downward
// in total dice, so we process states by decreasing total and never revisit one.
//
// Build: c++ -O3 -mcpu=native -o opt_bucket opt_bucket.cpp

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>

#include "biscuits.h"

static double V[NUM_STATES];    // V[state] = optimal expected remaining score
static float  Vf[NUM_STATES];   // float copy of V for the hot inner loop

// Solve V exactly, then make the float copy used by the hot inner loop.
static void solve() {
    solveV(V);
    for (int i = 0; i < 104; ++i) Vf[i] = (float)V[i];
}

// xorshift64 PRNG. Returns a penalty uniform on 0..sides-1 (Lemire, no modulo).
static inline uint64_t xorshift(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}
static inline int rollDie(uint64_t &s, int sides) {
    return (int)(((xorshift(s) >> 32) * (uint64_t)sides) >> 32);
}

static constexpr int CH = 1024;   // chunk: S6arr (13*CH*4 = 52KB) fits L1

// per-state game pools (SoA): rng state + accumulated score
static std::vector<uint64_t> bucketRng[104];
static std::vector<int>      bucketScore[104];

int main(int argc, char **argv) {
    long N = (argc > 1) ? atol(argv[1]) : 20000000;
    solve();
    printf("optimal V(12,1,1,1) = %.6f\n", V[stateIndex(12, 1, 1, 1)]);

    // seed the start bucket
    int START = stateIndex(12, 1, 1, 1);
    bucketRng[START].resize(N);
    bucketScore[START].assign(N, 0);
    for (long i = 0; i < N; ++i) {
        uint64_t s = (uint64_t)(i + 1) * 0x9E3779B97F4A7C15ULL;
        s ^= s >> 31;
        if (!s) s = 1;
        bucketRng[START][i] = s;
    }

    double sum = 0, sumsq = 0;
    long done = 0, rolls = 0;
    alignas(64) static float S6arr[13 * CH];
    alignas(64) static int   p8a[CH], p10a[CH], p12a[CH];
    alignas(64) static float best[CH];
    alignas(64) static int   bestc[CH];

    auto t0 = std::chrono::steady_clock::now();

    for (int totalDice = 15; totalDice >= 1; --totalDice) {
      for (int d6 = 0; d6 <= 12; ++d6)
      for (int d8 = 0; d8 <= 1; ++d8)
      for (int d10 = 0; d10 <= 1; ++d10)
      for (int d12 = 0; d12 <= 1; ++d12) {
        if (d6 + d8 + d10 + d12 != totalDice) continue;
        int S = stateIndex(d6, d8, d10, d12);
        long m = (long)bucketRng[S].size();
        if (!m) continue;
        uint64_t *RNG = bucketRng[S].data();
        int      *SC  = bucketScore[S].data();

        for (long off = 0; off < m; off += CH) {
          int n = (int)((m - off < CH) ? (m - off) : CH);

          // --- roll + per-game sorted prefix sums S6arr[k6*CH + i] ---
          for (int i = 0; i < n; ++i) {
            uint64_t s = RNG[off + i];
            int faceHist[6] = {0, 0, 0, 0, 0, 0};
            for (int j = 0; j < d6; ++j) faceHist[rollDie(s, 6)]++;
            int kept = 0;
            S6arr[0 * CH + i] = 0;
            for (int penalty = 5; penalty >= 0; --penalty)
              for (int c = 0; c < faceHist[penalty]; ++c) {
                S6arr[(kept + 1) * CH + i] = S6arr[kept * CH + i] + penalty;
                ++kept;
              }
            p8a[i]  = d8  ? rollDie(s, 8)  : 0;
            p10a[i] = d10 ? rollDie(s, 10) : 0;
            p12a[i] = d12 ? rollDie(s, 12) : 0;
            RNG[off + i] = s;
          }
          rolls += n;

          // --- argmax over comps, SIMD across games ---
          #pragma omp simd
          for (int i = 0; i < n; ++i) {
            best[i]  = -1e30f;
            bestc[i] = 0;
          }
          for (int keep8 = 0; keep8 <= d8; ++keep8)
          for (int keep10 = 0; keep10 <= d10; ++keep10)
          for (int keep12 = 0; keep12 <= d12; ++keep12) {
            int bigIdx  = keep8 * 4 + keep10 * 2 + keep12;
            int bigFull = (keep8 == d8 && keep10 == d10 && keep12 == d12);
            for (int keep6 = 0; keep6 <= d6; ++keep6) {
              if (bigFull && keep6 == d6) continue;       // skip illegal keep-everything
              float Vc = Vf[keep6 * 8 + bigIdx];
              int comp = keep6 * 8 + bigIdx;
              const float *s6 = &S6arr[keep6 * CH];
              #pragma omp simd
              for (int i = 0; i < n; ++i) {
                float v = s6[i] + (float)((keep8 ? p8a[i] : 0) + (keep10 ? p10a[i] : 0) + (keep12 ? p12a[i] : 0)) - Vc;
                int better = v > best[i];
                best[i]  = better ? v    : best[i];
                bestc[i] = better ? comp : bestc[i];
              }
            }
          }

          // --- transition: bank, score, scatter to new bucket ---
          const float *s6top = &S6arr[d6 * CH];
          for (int i = 0; i < n; ++i) {
            int c = bestc[i];
            int bestKeep6 = c >> 3, bigIdx = c & 7;
            int bestKeep8 = bigIdx >> 2, bestKeep10 = (bigIdx >> 1) & 1, bestKeep12 = bigIdx & 1;
            int banked = (int)(s6top[i] - S6arr[bestKeep6 * CH + i])
                + (bestKeep8 ? 0 : p8a[i]) + (bestKeep10 ? 0 : p10a[i]) + (bestKeep12 ? 0 : p12a[i]);
            int newScore = SC[off + i] + banked;
            int newState = stateIndex(bestKeep6, bestKeep8, bestKeep10, bestKeep12);
            if (newState == 0) {
              sum += newScore;
              sumsq += (double)newScore * newScore;
              ++done;
            } else {
              bucketRng[newState].push_back(RNG[off + i]);
              bucketScore[newState].push_back(newScore);
            }
          }
        }
        // free this bucket's pools
        std::vector<uint64_t>().swap(bucketRng[S]);
        std::vector<int>().swap(bucketScore[S]);
      }
    }

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mean = sum / done, sd = sqrt(sumsq / done - mean * mean);
    printf("BUCKETED-SIMD: %.2f s -> %.3f M games/s, %.1f M moves/s (%.2f rolls/game)\n",
           secs, done / secs / 1e6, rolls / secs / 1e6, (double)rolls / done);
    printf("mean = %.5f  (sd %.4f, optimal=%.5f, gap=%+.5f), games=%ld\n",
           mean, sd, V[stateIndex(12, 1, 1, 1)], mean - V[stateIndex(12, 1, 1, 1)], done);
    return 0;
}
