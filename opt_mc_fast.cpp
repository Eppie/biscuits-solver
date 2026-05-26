// Monte-Carlo validation of the PROVABLY OPTIMAL policy.
// Compute optimal V (DP), then play games choosing, each roll, the kept-set K
// that maximizes (sum of kept penalties) - V(K). The average score must
// converge to V(12,1,1,1) = 8.0879 if both the DP value and the greedy-on-V
// action are correct.
//
// This "fast" variant uses a xorshift128+ RNG, a face-histogram instead of a
// sort, and a branchless argmax with packed action indices in the policy loop.
//
// Build: c++ -O3 -mcpu=native -o opt_mc_fast opt_mc_fast.cpp

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>

#include "bitches.h"

static double V[NUM_STATES];    // V[state] = optimal expected remaining score

// ---- Fast RNG: xorshift128+ (single global stream, fixed seed) ----
static uint64_t rngState0 = 0x243F6A8885A308D3ULL;
static uint64_t rngState1 = 0x13198A2E03707344ULL;
static inline uint64_t xorshift(){
    uint64_t x = rngState0, y = rngState1;
    rngState0 = y;
    x ^= x << 23;
    rngState1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rngState1 + y;
}
// Uniform penalty in 0..sides-1.
static inline int rollDie(int sides){
    return (int)(((xorshift() >> 32) * (uint64_t)sides) >> 32);
}

int main(int argc, char** argv){
    long N = (argc > 1) ? atol(argv[1]) : 20000000;
    solveV(V);
    printf("optimal V(12,1,1,1) = %.6f  (target)\n", V[stateIndex(12, 1, 1, 1)]);

    // ---- Phase 2: Monte-Carlo replay of the greedy-on-V policy ----
    double sum = 0, sumSq = 0;
    long hist[64] = {0};
    double rolls = 0;
    auto t0 = std::chrono::steady_clock::now();
    for(long t = 0; t < N; ++t){
        int d6 = 12, d8 = 1, d10 = 1, d12 = 1;
        int score = 0;
        while(d6 + d8 + d10 + d12 > 0){
            rolls++;
            // roll d6 -> histogram of penalties, build top-k prefix sums (no sort).
            // faceHist[p] = number of d6 dice showing penalty p (0..5).
            int faceHist[6] = {0, 0, 0, 0, 0, 0};
            for(int j = 0; j < d6; ++j) faceHist[rollDie(6)]++;
            int topPenSum6[13];
            topPenSum6[0] = 0;
            {
                int k = 0;
                for(int penalty = 5; penalty >= 0; --penalty)
                    for(int c = 0; c < faceHist[penalty]; ++c){
                        topPenSum6[k+1] = topPenSum6[k] + penalty;
                        ++k;
                    }
            }
            int pen8  = d8  ? rollDie(8)  : 0;
            int pen10 = d10 ? rollDie(10) : 0;
            int pen12 = d12 ? rollDie(12) : 0;
            int totalPen = topPenSum6[d6]
                + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

            // greedy on V, BRANCHLESS: encode action as packed = k6*8 + bigPacked,
            // use conditional-moves; exclude illegal "keep everything" via a select.
            float best = -1e30f;
            int bestPacked = 0;
            for(int try8  = 0; try8  <= d8;  ++try8)
            for(int try10 = 0; try10 <= d10; ++try10)
            for(int try12 = 0; try12 <= d12; ++try12){
                int bigPacked = try8 * 4 + try10 * 2 + try12;
                float bigKept = (float)((try8 ? pen8 : 0) + (try10 ? pen10 : 0) + (try12 ? pen12 : 0));
                const double* Vp = V + bigPacked;
                int bigFull = (try8 == d8 && try10 == d10 && try12 == d12);
                for(int try6 = 0; try6 <= d6; ++try6){
                    float v = (float)topPenSum6[try6] + bigKept - (float)Vp[try6 * 8];
                    int illegal = bigFull & (try6 == d6);
                    v = illegal ? -1e30f : v;          // cmov
                    int packed = try6 * 8 + bigPacked;
                    int better = v > best;             // cmov pair below
                    best = better ? v : best;
                    bestPacked = better ? packed : bestPacked;
                }
            }
            int keep6 = bestPacked >> 3;
            int bidx = bestPacked & 7;
            int keep8 = bidx >> 2, keep10 = (bidx >> 1) & 1, keep12 = bidx & 1;
            int keptPen = topPenSum6[keep6]
                + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
            int banked = totalPen - keptPen;                 // banked penalty
            score += banked;
            d6 = keep6;
            d8 = keep8;
            d10 = keep10;
            d12 = keep12;
        }
        sum += score;
        sumSq += (double)score * score;
        if(score < 64) hist[score]++;
    }
    double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double mean = sum / N, sd = sqrt(sumSq / N - mean * mean);
    printf("MC loop (excl. DP solve): %.2f s  ->  %.3f M games/s,  %.1f M moves/s  (%.2f rolls/game)\n",
           secs, N/secs/1e6, rolls/secs/1e6, rolls/(double)N);
    printf("MC optimal-policy mean = %.5f  (sd %.4f, %ld games, stderr %.5f)\n",
           mean, sd, N, sd/sqrt((double)N));
    printf("difference from exact optimum = %+.5f\n", mean - V[stateIndex(12, 1, 1, 1)]);
    return 0;
}
