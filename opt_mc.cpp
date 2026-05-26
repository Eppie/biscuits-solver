// Monte-Carlo validation of the PROVABLY OPTIMAL policy.
// Compute optimal V (DP), then play games choosing, each roll, the kept-set K
// that maximizes (sum of kept penalties) - V(K). The average score must
// converge to V(12,1,1,1) = 8.0879 if both the DP value and the greedy-on-V
// action are correct.
//
// Build: c++ -O3 -mcpu=native -o opt_mc opt_mc.cpp

#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

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

int main(int argc, char** argv){
    long N = (argc > 1) ? atol(argv[1]) : 20000000;
    solve();
    printf("optimal V(12,1,1,1) = %.6f  (target)\n", V[stateIndex(12, 1, 1, 1)]);

    // ---- Phase 2: Monte-Carlo replay of the greedy-on-V policy ----
    std::mt19937_64 g(0xBADC0FFEE);
    double sum = 0, sumSq = 0;
    long hist[64] = {0};
    double rolls = 0;
    auto t0 = std::chrono::steady_clock::now();
    for(long t = 0; t < N; ++t){
        int d6 = 12, d8 = 1, d10 = 1, d12 = 1;
        int score = 0;
        while(d6 + d8 + d10 + d12 > 0){
            rolls++;
            // roll the d6 dice and build top-k prefix sums of their penalties
            int penalties[12];
            for(int j = 0; j < d6; ++j) penalties[j] = g() % 6;
            std::sort(penalties, penalties + d6, std::greater<int>());    // penalties desc
            int topPenSum6[13];
            topPenSum6[0] = 0;
            for(int k = 0; k < d6; ++k) topPenSum6[k+1] = topPenSum6[k] + penalties[k];

            int pen8  = d8  ? g() % 8  : 0;
            int pen10 = d10 ? g() % 10 : 0;
            int pen12 = d12 ? g() % 12 : 0;
            int totalPen = topPenSum6[d6]
                + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

            // greedy on V: pick kept composition maximizing kept_pen - V(K)
            double bestKeepValue = -1e300;
            int keep6 = 0, keep8 = 0, keep10 = 0, keep12 = 0;
            for(int try6  = 0; try6  <= d6;  ++try6)
            for(int try8  = 0; try8  <= d8;  ++try8)
            for(int try10 = 0; try10 <= d10; ++try10)
            for(int try12 = 0; try12 <= d12; ++try12){
                if(try6 == d6 && try8 == d8 && try10 == d10 && try12 == d12)
                    continue;
                double value = topPenSum6[try6]
                    + (try8 ? pen8 : 0) + (try10 ? pen10 : 0) + (try12 ? pen12 : 0)
                    - V[stateIndex(try6, try8, try10, try12)];
                if(value > bestKeepValue){
                    bestKeepValue = value;
                    keep6 = try6;
                    keep8 = try8;
                    keep10 = try10;
                    keep12 = try12;
                }
            }
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
