// biscuits (a dice game) — EXACT optimal policy via dynamic programming.
//
// This is the stand-alone driver for the value DP. The DP itself (solveV) and the
// state encoding live in biscuits.h, since most programs in this repo need them; see
// that header for the fully annotated algorithm. Here we just solve V and print it.
//
//   V(S) = sum over all roll outcomes  prob(outcome) * min over kept-sets K < S
//              [ (penalties of the dice you bank) + V(K) ]
//
// We enumerate every d6 face-composition with its exact multinomial weight (no Monte
// Carlo), so V is exact and the induced policy is provably optimal: each roll, keep
// the subset K maximising (sum of kept penalties) - V(K).
//
// Build: c++ -O3 -mcpu=native -o exact_dp exact_dp.cpp   (or -march=native on x86)

#include <cstdio>
#include <chrono>
#include "biscuits.h"

static double V[NUM_STATES];        // V[state] = optimal expected remaining score

int main(){
    auto startTime = std::chrono::steady_clock::now();

    solveV(V);

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - startTime).count();

    printf("EXACT optimal expected score V(12,1,1,1) = %.6f\n", V[stateIndex(12, 1, 1, 1)]);
    printf("(compute time %.2f s)\n\n", seconds);

    printf("Optimal V for selected states (d6,d8,d10,d12):\n");
    int show[][4] = {{12,1,1,1},{12,0,0,0},{6,1,1,1},{6,0,0,0},{3,0,0,0},{1,0,0,0},
                     {0,0,0,1},{0,0,1,0},{0,1,0,0},{1,1,1,1},{2,1,1,1}};
    for(auto& s : show)
        printf("  V(%2d,%d,%d,%d) = %.4f\n",
               s[0], s[1], s[2], s[3], V[stateIndex(s[0], s[1], s[2], s[3])]);
    return 0;
}
