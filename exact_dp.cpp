// bitches (a dice game) — EXACT optimal policy via dynamic programming.
//
// A game state is the multiset of dice still in hand:
//   d6                 : number of six-sided dice left, 0..12
//   d8, d10, d12       : whether that single big die is still in hand, 0 or 1
// That is 13 * 2 * 2 * 2 = 104 states. V(state) = optimal expected remaining
// score (lower is better), with V(0,0,0,0) = 0.
//
//   V(S) = sum over all roll outcomes  prob(outcome) * min over kept-sets K < S
//              [ (penalties of the dice you bank) + V(K) ]
//
// Within one roll, once you fix how MANY dice of each type to keep you keep the
// highest-penalty ones (bank the good dice, reroll the bad), because V(K) depends
// only on the composition of K, not which specific dice it contains. So the
// per-roll decision is a maximisation over <= 104 kept-compositions of
// (kept top-penalty sum) - V(K), and what you bank is (total penalty) minus that.
// We enumerate every d6 face-composition with its exact multinomial weight (no
// Monte Carlo), so V is exact and the induced policy is provably optimal.
//
// Build: c++ -O3 -mcpu=native -o exact_dp exact_dp.cpp   (or -march=native on x86)

#include <cstdio>
#include <cmath>
#include <chrono>

// Pack a state (d6 in 0..12; d8,d10,d12 in {0,1}) into a unique index 0..103.
static inline int stateIndex(int d6, int d8, int d10, int d12){
    return ((d6 * 2 + d8) * 2 + d10) * 2 + d12;
}

static double V[104];           // V[state] = optimal expected remaining score
static double factorial[13];    // factorial[n] = n!
static double pow6[13];         // pow6[n] = 6^n

int main(){
    factorial[0] = 1;
    for(int i = 1; i < 13; ++i) factorial[i] = factorial[i-1] * i;
    for(int n = 0; n < 13; ++n) pow6[n] = std::pow(6.0, n);

    auto startTime = std::chrono::steady_clock::now();

    V[stateIndex(0, 0, 0, 0)] = 0.0;    // empty hand: nothing left to score

    // Solve states in order of increasing dice count, so that every smaller
    // "kept" sub-state K has already been computed before we need it.
    for(int totalDice = 1; totalDice <= 15; ++totalDice){
      for(int d6 = 0; d6 <= 12; ++d6)
      for(int d8 = 0; d8 <= 1; ++d8)
      for(int d10 = 0; d10 <= 1; ++d10)
      for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice) continue;

        double expectedScore = 0.0;

        // Enumerate how the d6 dice land, as counts of each face value 1..6
        // (a multinomial composition summing to d6). The penalty of a d6 showing
        // face f is 6 - f, so face 1 -> penalty 5, ..., face 6 -> penalty 0.
        for(int face1 = 0; face1 <= d6; ++face1)
        for(int face2 = 0; face2 <= d6 - face1; ++face2)
        for(int face3 = 0; face3 <= d6 - face1 - face2; ++face3)
        for(int face4 = 0; face4 <= d6 - face1 - face2 - face3; ++face4)
        for(int face5 = 0; face5 <= d6 - face1 - face2 - face3 - face4; ++face5){
            int face6 = d6 - face1 - face2 - face3 - face4 - face5;

            // Probability of this exact face-count combination.
            double prob6 = (factorial[d6] /
                (factorial[face1] * factorial[face2] * factorial[face3] *
                 factorial[face4] * factorial[face5] * factorial[face6]))
                / pow6[d6];

            // topPenSum6[k] = sum of the k largest d6 penalties this roll
            // (penalty-5 dice first, then 4, ...). Keeping k of your d6 means
            // keeping the k highest-penalty ones, so this is what you would bank.
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

            // Each big die present shows one penalty, uniformly distributed.
            // Enumerate every possibility and weight by its probability.
            double prob8  = d8  ? 1.0 / 8  : 1.0;
            double prob10 = d10 ? 1.0 / 10 : 1.0;
            double prob12 = d12 ? 1.0 / 12 : 1.0;

            for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
            for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
            for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
                // Penalty contributed by each big die if you keep it (index 1)
                // versus reroll it (index 0).
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
                        continue;   // banking nothing is not allowed
                    double value = topPenSum6[keep6] + keep8Pen[keep8]
                        + keep10Pen[keep10] + keep12Pen[keep12]
                        - V[stateIndex(keep6, keep8, keep10, keep12)];
                    if(value > bestKeepValue) bestKeepValue = value;
                }

                double minBanked = totalPen - bestKeepValue;
                expectedScore += prob6 * prob8 * prob10 * prob12 * minBanked;
            }
        }
        V[stateIndex(d6, d8, d10, d12)] = expectedScore;
      }
    }

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
