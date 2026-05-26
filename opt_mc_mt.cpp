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

// Pack a state (d6 in 0..12; d8,d10,d12 in {0,1}) into a unique index 0..103.
static inline int stateIndex(int d6, int d8, int d10, int d12){
    return ((d6 * 2 + d8) * 2 + d10) * 2 + d12;
}

static double V[104];           // V[state] = optimal expected remaining score
static double factorial[13];    // factorial[n] = n!
static double pow6[13];         // pow6[n] = 6^n

// ---- Phase 1: exact DP solve of the value function V ----
static void solve(){
    factorial[0] = 1;
    for(int i = 1; i < 13; ++i) factorial[i] = factorial[i-1] * i;
    for(int n = 0; n < 13; ++n) pow6[n] = pow(6.0, n);

    V[stateIndex(0, 0, 0, 0)] = 0;

    // Solve states in order of increasing dice count, so that every smaller
    // "kept" sub-state K has already been computed before we need it.
    for(int totalDice = 1; totalDice <= 15; ++totalDice){
      for(int d6 = 0; d6 <= 12; ++d6)
      for(int d8 = 0; d8 <= 1; ++d8)
      for(int d10 = 0; d10 <= 1; ++d10)
      for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice) continue;

        double expectedScore = 0;

        // Enumerate how the d6 dice land, as counts of each face value 1..6.
        for(int face1 = 0; face1 <= d6; ++face1)
        for(int face2 = 0; face2 <= d6 - face1; ++face2)
        for(int face3 = 0; face3 <= d6 - face1 - face2; ++face3)
        for(int face4 = 0; face4 <= d6 - face1 - face2 - face3; ++face4)
        for(int face5 = 0; face5 <= d6 - face1 - face2 - face3 - face4; ++face5){
            int face6 = d6 - face1 - face2 - face3 - face4 - face5;

            double prob6 = (factorial[d6] /
                (factorial[face1] * factorial[face2] * factorial[face3] *
                 factorial[face4] * factorial[face5] * factorial[face6]))
                / pow6[d6];

            // topPenSum6[k] = sum of the k largest d6 penalties this roll.
            double topPenSum6[13];
            topPenSum6[0] = 0;
            {
                int faceCount[6] = {face1, face2, face3, face4, face5, face6};
                int kept = 0;
                for(int penalty = 5, f = 0; penalty >= 0; --penalty, ++f)
                    for(int j = 0; j < faceCount[f]; ++j){
                        topPenSum6[kept + 1] = topPenSum6[kept] + penalty;
                        ++kept;
                    }
            }
            double totalPen6 = topPenSum6[d6];

            double prob8  = d8  ? 1.0 / 8  : 1;
            double prob10 = d10 ? 1.0 / 10 : 1;
            double prob12 = d12 ? 1.0 / 12 : 1;

            for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
            for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
            for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
                // Penalty per big die if kept (index 1) vs rerolled (index 0).
                double keep8Pen[2]  = {0, (double)pen8};
                double keep10Pen[2] = {0, (double)pen10};
                double keep12Pen[2] = {0, (double)pen12};

                double totalPen = totalPen6
                    + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

                // Pick the kept sub-hand K (a proper subset; you must bank >= 1)
                // that maximises (penalties you bank) - V(K).
                double bestKeepValue = -1e300;
                for(int keep6  = 0; keep6  <= d6;  ++keep6)
                for(int keep8  = 0; keep8  <= d8;  ++keep8)
                for(int keep10 = 0; keep10 <= d10; ++keep10)
                for(int keep12 = 0; keep12 <= d12; ++keep12){
                    if(keep6 == d6 && keep8 == d8 && keep10 == d10 && keep12 == d12)
                        continue;
                    double value = topPenSum6[keep6] + keep8Pen[keep8]
                        + keep10Pen[keep10] + keep12Pen[keep12]
                        - V[stateIndex(keep6, keep8, keep10, keep12)];
                    if(value > bestKeepValue) bestKeepValue = value;
                }

                expectedScore += prob6 * prob8 * prob10 * prob12
                    * (totalPen - bestKeepValue);
            }
        }
        V[stateIndex(d6, d8, d10, d12)] = expectedScore;
      }
    }
}

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
    solve();
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
