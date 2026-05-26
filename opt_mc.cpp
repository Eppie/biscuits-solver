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

#include "bitches.h"

static double V[NUM_STATES];    // V[state] = optimal expected remaining score

int main(int argc, char** argv){
    long N = (argc > 1) ? atol(argv[1]) : 20000000;
    solveV(V);
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
