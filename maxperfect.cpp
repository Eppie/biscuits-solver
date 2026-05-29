// Policy that MAXIMIZES P(perfect game), and its probability.
// To finish at 0 every banked die must be a max face (penalty 0). So you only
// ever bank max-faced dice; the choice each roll is WHICH nonempty subset of the
// max-faced dice to bank (bank fewer => keep more dice in hand to generate maxes
// next roll, but more rolls to survive). If no die shows its max you are forced
// to bank a non-max => run ends imperfect.
//
//   pPerfectMax(S) = E_Z[ Z=empty ? 0 : max over nonempty B<=Z of pPerfectMax(S - B) ]
//
// where Z = multiset of dice showing their max this roll (independent per die).
//
// A game state is the multiset of dice still in hand:
//   d6                 : number of six-sided dice left, 0..12
//   d8, d10, d12       : whether that single big die is still in hand, 0 or 1
// That is 13 * 2 * 2 * 2 = 104 states.
//
// Build: c++ -O3 -mcpu=native -o maxperfect maxperfect.cpp

#include <cstdio>
#include <cmath>
#include <random>

#include "biscuits.h"

static double pPerfectMax[104];      // pPerfectMax[state] = max achievable P(perfect)
static double binomTable[13][13];    // binomTable[n][k]   = binomial coefficient C(n,k)
static const double pShowMax[4] = {1.0/6, 1.0/8, 1.0/10, 1.0/12}; // P(die shows its max)

static inline double binom(int n, int k, double q){
    return binomTable[n][k] * pow(q, k) * pow(1 - q, n - k);
}

int main(int argc, char** argv){
    long N = (argc > 1) ? atol(argv[1]) : 50000000;
    for(int n = 0; n < 13; ++n){
        binomTable[n][0] = 1;
        for(int k = 1; k <= n; ++k) binomTable[n][k] = binomTable[n-1][k-1] + binomTable[n-1][k];
    }

    pPerfectMax[stateIndex(0, 0, 0, 0)] = 1.0;
    bool alwaysBankAll = true;
    for(int totalDice = 1; totalDice <= 15; ++totalDice){
      for(int d6 = 0; d6 <= 12; ++d6)
      for(int d8 = 0; d8 <= 1; ++d8)
      for(int d10 = 0; d10 <= 1; ++d10)
      for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice) continue;

        double acc = 0;

        // Sum over Z = (maxShown6, maxShown8, maxShown10, maxShown12) dice showing max.
        for(int maxShown6 = 0; maxShown6 <= d6; ++maxShown6)
        for(int maxShown8 = 0; maxShown8 <= d8; ++maxShown8)
        for(int maxShown10 = 0; maxShown10 <= d10; ++maxShown10)
        for(int maxShown12 = 0; maxShown12 <= d12; ++maxShown12){
          if(maxShown6 + maxShown8 + maxShown10 + maxShown12 == 0)
              continue;   // no max shown -> forced imperfect (0)

          double pz = binom(d6, maxShown6, pShowMax[0]) * binom(d8, maxShown8, pShowMax[1])
              * binom(d10, maxShown10, pShowMax[2]) * binom(d12, maxShown12, pShowMax[3]);

          // Choose nonempty banked subset B <= Z maximizing pPerfectMax[S - B].
          double bestChild = -1;
          int bank6 = 0, bank8 = 0, bank10 = 0, bank12 = 0;
          for(int b6 = 0; b6 <= maxShown6; ++b6)
          for(int b8 = 0; b8 <= maxShown8; ++b8)
          for(int b10 = 0; b10 <= maxShown10; ++b10)
          for(int b12 = 0; b12 <= maxShown12; ++b12){
            if(b6 + b8 + b10 + b12 == 0) continue;
            double child = pPerfectMax[stateIndex(d6 - b6, d8 - b8, d10 - b10, d12 - b12)];
            if(child > bestChild){
                bestChild = child;
                bank6 = b6;
                bank8 = b8;
                bank10 = b10;
                bank12 = b12;
            }
          }
          if(!(bank6 == maxShown6 && bank8 == maxShown8 && bank10 == maxShown10 && bank12 == maxShown12))
              alwaysBankAll = false;   // kept a maxed die
          acc += pz * bestChild;
        }
        pPerfectMax[stateIndex(d6, d8, d10, d12)] = acc;
      }
    }

    double p = pPerfectMax[stateIndex(12, 1, 1, 1)];
    printf("MAX P(perfect) = %.6e  = 1 in %.0f  (%.4f%%)\n", p, 1.0/p, 100*p);
    printf("optimal max-perfect policy is 'bank EVERY max-faced die each roll': %s\n",
           alwaysBankAll ? "YES" : "NO (sometimes keep a max-faced die)");
    printf("(for reference, under expected-optimal play it was 1.613e-3 = 1 in 620)\n");

    // MC of the ACTUAL optimal max-perfect policy (choose bank-subset via pPerfectMax).
    std::mt19937_64 rng(0xFEED);
    long zeros = 0;
    for(long t = 0; t < N; ++t){
        int d6 = 12, d8 = 1, d10 = 1, d12 = 1;
        bool perfect = true;
        while(d6 + d8 + d10 + d12 > 0){
            int maxShown6 = 0;
            for(int j = 0; j < d6; ++j) if(rng() % 6 == 5) maxShown6++;
            int maxShown8  = d8  ? (rng() % 8  == 7)  : 0;
            int maxShown10 = d10 ? (rng() % 10 == 9)  : 0;
            int maxShown12 = d12 ? (rng() % 12 == 11) : 0;
            if(maxShown6 + maxShown8 + maxShown10 + maxShown12 == 0){
                perfect = false;
                break;
            }
            double bestChild = -1;
            int bank6 = 0, bank8 = 0, bank10 = 0, bank12 = 0;
            for(int b6 = 0; b6 <= maxShown6; ++b6)
            for(int b8 = 0; b8 <= maxShown8; ++b8)
            for(int b10 = 0; b10 <= maxShown10; ++b10)
            for(int b12 = 0; b12 <= maxShown12; ++b12){
              if(b6 + b8 + b10 + b12 == 0) continue;
              double child = pPerfectMax[stateIndex(d6 - b6, d8 - b8, d10 - b10, d12 - b12)];
              if(child > bestChild){
                  bestChild = child;
                  bank6 = b6;
                  bank8 = b8;
                  bank10 = b10;
                  bank12 = b12;
              }
            }
            d6 -= bank6;
            d8 -= bank8;
            d10 -= bank10;
            d12 -= bank12;
        }
        if(perfect) zeros++;
    }
    printf("MC (optimal max-perfect policy) P(perfect) = %.6e  (%ld/%ld)\n", (double)zeros/N, zeros, N);
    return 0;
}
