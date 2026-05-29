// biscuits (a dice game) — shared core used by every solver/simulator in this repo.
//
// The game uses 15 dice: 12 d6, one d8, one d10, one d12. Each roll you reroll the
// dice in hand and must bank at least one; a banked die scores penalty = max_face -
// shown_face (0 if banked at its max). Final score is the sum of penalties, lowest
// wins. See README.md for the full analysis.
//
// A game state is the multiset of dice still in hand:
//   d6                 : number of six-sided dice left, 0..12
//   d8, d10, d12       : whether that single big die is still in hand, 0 or 1
// That is 13 * 2 * 2 * 2 = 104 states. This header provides the pieces that were
// otherwise copy-pasted across files: the state index, the optimal value function V
// and the optimal per-roll action it induces, and the d6 top-penalty prefix sums.
//
// Everything here is `static inline` so each single-file program can include it
// without link issues (each .cpp builds into its own binary).

#ifndef BISCUITS_H
#define BISCUITS_H

#include <cmath>

enum { NUM_STATES = 104 };

// Pack a state (d6 in 0..12; d8,d10,d12 in {0,1}) into a unique index 0..103.
static inline int stateIndex(int d6, int d8, int d10, int d12){
    return ((d6 * 2 + d8) * 2 + d10) * 2 + d12;
}

// topPenSum6[k] = sum of the k largest d6 penalties given faceCount[f] = number of
// d6 showing face value f+1 (penalty 5,4,3,2,1,0 for f = 0..5). Keeping k of your
// d6 means keeping the k highest-penalty ones, so this is what you would bank.
static inline void buildTopPenSums(const int faceCount[6], double topPenSum6[13]){
    topPenSum6[0] = 0;
    int kept = 0;
    for(int penalty = 5, f = 0; penalty >= 0; --penalty, ++f)
        for(int j = 0; j < faceCount[f]; ++j){
            topPenSum6[kept + 1] = topPenSum6[kept] + penalty;
            ++kept;
        }
}

// Solve the EXACT expected-optimal value function into V (size 104):
//   V(state) = optimal expected remaining score (lower is better), V(0,0,0,0) = 0.
// We enumerate every d6 face-composition with its exact multinomial weight and every
// big-die outcome (no Monte Carlo), so V is exact and the induced policy is provably
// optimal. States are solved in increasing dice count so every smaller "kept"
// sub-state is already known. This is the canonical DP; see exact_dp.cpp for the
// stand-alone, fully annotated version.
static inline void solveV(double* V){
    double factorial[13];
    factorial[0] = 1;
    for(int i = 1; i < 13; ++i) factorial[i] = factorial[i-1] * i;
    double pow6[13];
    for(int n = 0; n < 13; ++n) pow6[n] = std::pow(6.0, n);

    V[stateIndex(0, 0, 0, 0)] = 0.0;

    for(int totalDice = 1; totalDice <= 15; ++totalDice){
      for(int d6 = 0; d6 <= 12; ++d6)
      for(int d8 = 0; d8 <= 1; ++d8)
      for(int d10 = 0; d10 <= 1; ++d10)
      for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice) continue;

        double expectedScore = 0.0;

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

            int faceCount[6] = {face1, face2, face3, face4, face5, face6};
            double topPenSum6[13];
            buildTopPenSums(faceCount, topPenSum6);
            double totalPen6 = topPenSum6[d6];

            double prob8  = d8  ? 1.0 / 8  : 1.0;
            double prob10 = d10 ? 1.0 / 10 : 1.0;
            double prob12 = d12 ? 1.0 / 12 : 1.0;

            for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
            for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
            for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
                double keep8Pen[2]  = {0, (double)pen8};
                double keep10Pen[2] = {0, (double)pen10};
                double keep12Pen[2] = {0, (double)pen12};

                double totalPen = totalPen6
                    + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

                double bestKeepValue = -1e300;
                for(int try6  = 0; try6  <= d6;  ++try6)
                for(int try8  = 0; try8  <= d8;  ++try8)
                for(int try10 = 0; try10 <= d10; ++try10)
                for(int try12 = 0; try12 <= d12; ++try12){
                    if(try6 == d6 && try8 == d8 && try10 == d10 && try12 == d12)
                        continue;   // banking nothing is not allowed
                    double value = topPenSum6[try6] + keep8Pen[try8]
                        + keep10Pen[try10] + keep12Pen[try12]
                        - V[stateIndex(try6, try8, try10, try12)];
                    if(value > bestKeepValue) bestKeepValue = value;
                }

                double minBanked = totalPen - bestKeepValue;
                expectedScore += prob6 * prob8 * prob10 * prob12 * minBanked;
            }
        }
        V[stateIndex(d6, d8, d10, d12)] = expectedScore;
      }
    }
}

// The expected-optimal action for one roll: choose how many of each die type to keep
// so as to maximise (penalties you bank) - V(kept hand). The d6 penalties available
// are summarised by topPenSum6 (see buildTopPenSums); pen8/pen10/pen12 are the
// penalties the big dice show this roll. Canonical tie-break: the first maximiser
// found (fewest dice kept = bank more), matching every Monte-Carlo simulator here.
static inline void optimalKeep(const double* V, int d6, int d8, int d10, int d12,
                               const double* topPenSum6, int pen8, int pen10, int pen12,
                               int& keep6, int& keep8, int& keep10, int& keep12){
    double keep8Pen[2]  = {0, (double)pen8};
    double keep10Pen[2] = {0, (double)pen10};
    double keep12Pen[2] = {0, (double)pen12};

    double bestKeepValue = -1e300;
    keep6 = keep8 = keep10 = keep12 = 0;
    for(int try6  = 0; try6  <= d6;  ++try6)
    for(int try8  = 0; try8  <= d8;  ++try8)
    for(int try10 = 0; try10 <= d10; ++try10)
    for(int try12 = 0; try12 <= d12; ++try12){
        if(try6 == d6 && try8 == d8 && try10 == d10 && try12 == d12)
            continue;
        double value = topPenSum6[try6] + keep8Pen[try8]
            + keep10Pen[try10] + keep12Pen[try12]
            - V[stateIndex(try6, try8, try10, try12)];
        if(value > bestKeepValue){
            bestKeepValue = value;
            keep6 = try6; keep8 = try8; keep10 = try10; keep12 = try12;
        }
    }
}

#endif // BISCUITS_H
