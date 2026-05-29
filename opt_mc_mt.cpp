// Monte-Carlo validation of the PROVABLY OPTIMAL policy.
// Compute optimal V (DP), then play games choosing, each roll, the kept-set K
// that maximizes (sum of kept penalties) - V(K). The average score must
// converge to V(12,1,1,1) = 8.0879 if both the DP value and the greedy-on-V
// action are correct.
//
// Multi-threaded variant: split N games across worker threads, each with its
// own seeded xorshift128+ stream, then reduce per-thread Stat accumulators.
//
// Build: c++ -O3 -mcpu=native -pthread -o opt_mc_mt opt_mc_mt.cpp

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <thread>
#include <vector>

#include "biscuits.h"

static double V[NUM_STATES];    // V[state] = optimal expected remaining score

// ---- Fast RNG: xorshift128+ (per-thread stream via passed-in state) ----
static inline uint64_t xorshift(uint64_t& state0, uint64_t& state1){
    uint64_t x = state0, y = state1;
    state0 = y;
    x ^= x << 23;
    state1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return state1 + y;
}
// Uniform penalty in 0..sides-1.
static inline int rollDie(uint64_t& state0, uint64_t& state1, int sides){
    return (int)(((xorshift(state0, state1) >> 32) * (uint64_t)sides) >> 32);
}

// Per-thread Monte-Carlo accumulators (reduced after join).
struct Stat{
    double sum = 0;        // sum of scores
    double sumSq = 0;      // sum of squared scores
    double rolls = 0;      // total rolls played
    long hist[64] = {0};   // score histogram
};

// ---- Phase 2 worker: play `games` games on a private RNG stream ----
static void worker(long games, uint64_t seed, Stat& st){
    uint64_t rngState0 = seed * 0x9E3779B97F4A7C15ULL + 1;
    uint64_t rngState1 = seed * 0xD1B54A32D192ED03ULL + 0x9E3779B9ULL;
    for(int w = 0; w < 4; ++w) xorshift(rngState0, rngState1);
    for(long t = 0; t < games; ++t){
        int d6 = 12, d8 = 1, d10 = 1, d12 = 1;
        int score = 0;
        while(d6 + d8 + d10 + d12 > 0){
            st.rolls++;
            // roll d6 -> histogram of penalties, build top-k prefix sums (no sort).
            // faceHist[p] = number of d6 dice showing penalty p (0..5).
            int faceHist[6] = {0, 0, 0, 0, 0, 0};
            for(int j = 0; j < d6; ++j) faceHist[rollDie(rngState0, rngState1, 6)]++;
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
            int pen8  = d8  ? rollDie(rngState0, rngState1, 8)  : 0;
            int pen10 = d10 ? rollDie(rngState0, rngState1, 10) : 0;
            int pen12 = d12 ? rollDie(rngState0, rngState1, 12) : 0;
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
                    v = (bigFull & (try6 == d6)) ? -1e30f : v;
                    int packed = try6 * 8 + bigPacked;
                    int better = v > best;
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
        st.sum += score;
        st.sumSq += (double)score * score;
        if(score < 64) st.hist[score]++;
    }
}

int main(int argc, char** argv){
    long N = (argc > 1) ? atol(argv[1]) : 20000000;
    solveV(V);
    printf("optimal V(12,1,1,1) = %.6f  (target)\n", V[stateIndex(12, 1, 1, 1)]);

    unsigned nth = (argc > 2) ? (unsigned)atoi(argv[2]) : std::thread::hardware_concurrency();
    if(!nth) nth = 1;
    std::vector<Stat> sts(nth);
    std::vector<std::thread> th;
    auto t0 = std::chrono::steady_clock::now();
    long per = N / nth;
    for(unsigned i = 0; i < nth; ++i){
        long g = (i == nth - 1) ? (N - per * (nth - 1)) : per;
        th.emplace_back(worker, g, (uint64_t)i + 1, std::ref(sts[i]));
    }
    for(auto& x : th) x.join();

    // Reduce per-thread stats in thread order (order-sensitive: identical
    // thread count must reproduce identical output).
    double sum = 0, sumSq = 0, rolls = 0;
    long hist[64] = {0};
    for(auto& x : sts){
        sum += x.sum;
        sumSq += x.sumSq;
        rolls += x.rolls;
        for(int i = 0; i < 64; ++i) hist[i] += x.hist[i];
    }
    double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double mean = sum / N, sd = sqrt(sumSq / N - mean * mean);
    printf("MC loop (excl. DP solve): %.2f s  ->  %.3f M games/s,  %.1f M moves/s  (%.2f rolls/game)\n",
           secs, N/secs/1e6, rolls/secs/1e6, rolls/(double)N);
    printf("threads: %u\n", nth);
    printf("MC optimal-policy mean = %.5f  (sd %.4f, %ld games, stderr %.5f)\n",
           mean, sd, N, sd/sqrt((double)N));
    printf("difference from exact optimum = %+.5f\n", mean - V[stateIndex(12, 1, 1, 1)]);
    return 0;
}
