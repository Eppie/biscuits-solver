// EXACT expected score of the THRESHOLD (Fetterman Blackjack-table) strategy,
// computed by the same 104-state DP but with the FIXED policy instead of the
// optimal max. Zero RNG -> if this equals our Monte Carlo (8.485), the sim is
// unbiased; if it equals Fetterman's 8.53, our policy/RNG has a skew.
//
// Policy: with `totalDice` dice in hand, bank every die whose penalty <=
// THR[size][totalDice]; if none qualify, bank the single die that "differs from
// its cutoff by the least" (min penalty - THR). THR[s][D] = floor(W[s][D-1])
// matches his Fig 7 exactly.
//
// Build: c++ -O3 -mcpu=native -o thr_dp thr_dp.cpp

#include <cstdio>
#include <cmath>
#include <chrono>
#include "bitches.h"

static double V[NUM_STATES];    // V[state] = expected remaining score under the threshold policy
static double factorial[13];    // factorial[n] = n!
static int THR[13][16];         // THR[size][dice left] = bank a die iff its penalty <= this

int main(){
    // ---------- build the threshold table ----------
    // For each die size s, W[m] is the expected penalty of the best single die
    // out of m rolled dice (Fetterman's recurrence). The bank cutoff with D dice
    // left is floor(W[D-1]); with <= 1 die left there is no cutoff (bank it).
    for(int s : {6, 8, 10, 12}){
        double W[16];
        W[1] = (s - 1) / 2.0;
        for(int m = 2; m <= 15; ++m){
            double c = W[m - 1];
            double a = 0;
            for(int p = 0; p < s; ++p) a += (p < c ? p : c);
            W[m] = a / s;
        }
        for(int D = 0; D <= 15; ++D) THR[s][D] = (D <= 1) ? s : (int)W[D - 1];
    }

    factorial[0] = 1;
    for(int i = 1; i < 13; ++i) factorial[i] = factorial[i - 1] * i;
    double pow6[13];
    for(int n = 0; n < 13; ++n) pow6[n] = pow(6.0, n);

    auto startTime = std::chrono::steady_clock::now();
    V[stateIndex(0, 0, 0, 0)] = 0.0;

    // Solve states in order of increasing dice count, so every "kept" sub-state
    // has already been computed before we need it.
    for(int totalDice = 1; totalDice <= 15; ++totalDice){
      for(int d6 = 0; d6 <= 12; ++d6)
      for(int d8 = 0; d8 <= 1; ++d8)
      for(int d10 = 0; d10 <= 1; ++d10)
      for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice) continue;

        // Cutoffs that apply to this state (one per die size).
        const int T6  = THR[6][totalDice];
        const int T8  = THR[8][totalDice];
        const int T10 = THR[10][totalDice];
        const int T12 = THR[12][totalDice];
        double expectedScore = 0.0;

        // Enumerate how the d6 dice land, as counts of each face value 1..6.
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
            int faceCount[6] = {face1, face2, face3, face4, face5, face6};   // counts of d6 penalty 5,4,3,2,1,0

            // d6 banked under threshold: those with penalty <= T6 (low-pen dice).
            int kept6 = 0;        // kept = high-penalty d6 (pen > T6)
            double keptPen6 = 0;
            int lowPen6 = 99;     // smallest d6 penalty present (for the forced case)
            for(int c = 0, pen = 5; c < 6; ++c, --pen){
                if(faceCount[c] > 0 && pen < lowPen6) lowPen6 = pen;
                if(pen > T6){
                    kept6 += faceCount[c];
                    keptPen6 += (double)pen * faceCount[c];
                }
            }

            // Each big die present shows one penalty, uniformly distributed.
            double prob8  = d8  ? 1.0 / 8  : 1.0;
            double prob10 = d10 ? 1.0 / 10 : 1.0;
            double prob12 = d12 ? 1.0 / 12 : 1.0;
            for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
            for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
            for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
                // big dice kept iff penalty > threshold
                int keep8  = (d8  && pen8  > T8 ) ? 1 : 0;
                int keep10 = (d10 && pen10 > T10) ? 1 : 0;
                int keep12 = (d12 && pen12 > T12) ? 1 : 0;
                double keptPen = keptPen6;
                if(keep8)  keptPen += pen8;
                if(keep10) keptPen += pen10;
                if(keep12) keptPen += pen12;

                // total penalty over all dice this roll
                double totalPen = 0;
                for(int c = 0, pen = 5; c < 6; ++c, --pen) totalPen += (double)pen * faceCount[c];
                if(d8)  totalPen += pen8;
                if(d10) totalPen += pen10;
                if(d12) totalPen += pen12;

                int keep6 = kept6;
                int K6 = keep6, K8 = keep8, K10 = keep10, K12 = keep12;
                double banked = totalPen - keptPen;   // banked = low-penalty dice

                // Forced: nothing qualified to bank (kept everything) -> bank the
                // single die with min (penalty - its threshold).
                if(K6 == d6 && K8 == d8 && K10 == d10 && K12 == d12){
                    double bestDiff = 1e300;
                    int removeSize = 0;       // which die size to bank: 6/8/10/12
                    double removePen = 0;
                    if(d6){
                        double diff = lowPen6 - T6;
                        if(diff < bestDiff){ bestDiff = diff; removeSize = 6;  removePen = lowPen6; }
                    }
                    if(d8){
                        double diff = pen8 - T8;
                        if(diff < bestDiff){ bestDiff = diff; removeSize = 8;  removePen = pen8; }
                    }
                    if(d10){
                        double diff = pen10 - T10;
                        if(diff < bestDiff){ bestDiff = diff; removeSize = 10; removePen = pen10; }
                    }
                    if(d12){
                        double diff = pen12 - T12;
                        if(diff < bestDiff){ bestDiff = diff; removeSize = 12; removePen = pen12; }
                    }
                    if(removeSize == 6)       K6--;
                    else if(removeSize == 8)  K8--;
                    else if(removeSize == 10) K10--;
                    else                      K12--;
                    banked = removePen;
                }

                double minBanked = banked + V[stateIndex(K6, K8, K10, K12)];
                expectedScore += prob6 * prob8 * prob10 * prob12 * minBanked;
            }
        }
        V[stateIndex(d6, d8, d10, d12)] = expectedScore;
      }
    }

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - startTime).count();
    printf("EXACT threshold-strategy expected score V(12,1,1,1) = %.6f\n", V[stateIndex(12, 1, 1, 1)]);
    printf("(our Monte Carlo was 8.485; Fetterman reported 8.53; compute %.2f s)\n",
           seconds);
    return 0;
}
