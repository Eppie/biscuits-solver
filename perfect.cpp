// P(perfect game, i.e. score 0) UNDER THE EXPECTED-OPTIMAL POLICY.
// A perfect game needs every banked die at its max face (penalty 0). The
// optimal (expected-minimizing) policy sometimes banks a low-but-nonzero die,
// which ends any perfect run. We compute pPerfect[S] = probability of finishing
// at 0 from state S while playing the optimal action each roll, exactly by
// enumeration, and validate with Monte Carlo.
//
// A game state is the multiset of dice still in hand:
//   d6                 : number of six-sided dice left, 0..12
//   d8, d10, d12       : whether that single big die is still in hand, 0 or 1
// That is 13 * 2 * 2 * 2 = 104 states.
//
// Build: c++ -O3 -mcpu=native -o perfect perfect.cpp

#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>

// Pack a state (d6 in 0..12; d8,d10,d12 in {0,1}) into a unique index 0..103.
static inline int stateIndex(int d6, int d8, int d10, int d12){
    return ((d6 * 2 + d8) * 2 + d10) * 2 + d12;
}

static double V[104];           // V[state]        = optimal expected remaining score
static double pPerfect[104];    // pPerfect[state] = P(finish at 0) under optimal play
static double factorial[13];    // factorial[n]    = n!
static double pow6[13];         // pow6[n]         = 6^n

// Solve the expected-score value function V exactly (same DP as exact_dp.cpp).
static void solveV(){
    factorial[0] = 1;
    for(int i = 1; i < 13; ++i) factorial[i] = factorial[i-1] * i;
    for(int n = 0; n < 13; ++n) pow6[n] = pow(6.0, n);

    V[stateIndex(0, 0, 0, 0)] = 0;

    // Solve states in order of increasing dice count so every smaller kept
    // sub-state is already computed before we need it.
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
              double keep8Pen[2]  = {0, (double)pen8};
              double keep10Pen[2] = {0, (double)pen10};
              double keep12Pen[2] = {0, (double)pen12};

              double totalPen = totalPen6
                  + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

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

              expectedScore += prob6 * prob8 * prob10 * prob12 * (totalPen - bestKeepValue);
          }
        }
        V[stateIndex(d6, d8, d10, d12)] = expectedScore;
      }
    }
}

// Optimal kept composition for a given roll (canonical tie-break: prefer the
// first found = fewer kept = bank more, matching the MC simulator).
static inline void optAction(int d6, int d8, int d10, int d12, const double* topPenSum6,
                             int pen8, int pen10, int pen12,
                             int& keep6, int& keep8, int& keep10, int& keep12){
    double keep8Pen[2]  = {0, (double)pen8};
    double keep10Pen[2] = {0, (double)pen10};
    double keep12Pen[2] = {0, (double)pen12};
    double bestKeepValue = -1e300;
    keep6 = keep8 = keep10 = keep12 = 0;
    for(int k6 = 0; k6 <= d6; ++k6)
    for(int k8 = 0; k8 <= d8; ++k8)
    for(int k10 = 0; k10 <= d10; ++k10)
    for(int k12 = 0; k12 <= d12; ++k12){
        if(k6 == d6 && k8 == d8 && k10 == d10 && k12 == d12) continue;
        double value = topPenSum6[k6] + keep8Pen[k8] + keep10Pen[k10] + keep12Pen[k12]
            - V[stateIndex(k6, k8, k10, k12)];
        if(value > bestKeepValue){
            bestKeepValue = value;
            keep6 = k6;
            keep8 = k8;
            keep10 = k10;
            keep12 = k12;
        }
    }
}

// Probability of a perfect (score-0) finish while always playing the optimal action.
static void solveP1(){
    pPerfect[stateIndex(0, 0, 0, 0)] = 1.0;
    for(int totalDice = 1; totalDice <= 15; ++totalDice){
      for(int d6 = 0; d6 <= 12; ++d6)
      for(int d8 = 0; d8 <= 1; ++d8)
      for(int d10 = 0; d10 <= 1; ++d10)
      for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice) continue;

        double acc = 0;

        for(int face1 = 0; face1 <= d6; ++face1)
        for(int face2 = 0; face2 <= d6 - face1; ++face2)
        for(int face3 = 0; face3 <= d6 - face1 - face2; ++face3)
        for(int face4 = 0; face4 <= d6 - face1 - face2 - face3; ++face4)
        for(int face5 = 0; face5 <= d6 - face1 - face2 - face3 - face4; ++face5){
          int face6 = d6 - face1 - face2 - face3 - face4 - face5;   // face6 = # of penalty-0 d6

          double prob6 = (factorial[d6] /
              (factorial[face1] * factorial[face2] * factorial[face3] *
               factorial[face4] * factorial[face5] * factorial[face6]))
              / pow6[d6];

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

          double prob8  = d8  ? 1.0 / 8  : 1;
          double prob10 = d10 ? 1.0 / 10 : 1;
          double prob12 = d12 ? 1.0 / 12 : 1;

          for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
          for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
          for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
              int keep6, keep8, keep10, keep12;
              optAction(d6, d8, d10, d12, topPenSum6, pen8, pen10, pen12,
                        keep6, keep8, keep10, keep12);

              // Banked all penalty-0? d6: banked count (d6-keep6) must be <= #zeros (face6).
              bool ok = (d6 - keep6) <= face6;
              if(d8  && keep8  == 0 && pen8  != 0) ok = false;   // a nonzero d8 was banked
              if(d10 && keep10 == 0 && pen10 != 0) ok = false;
              if(d12 && keep12 == 0 && pen12 != 0) ok = false;
              if(ok) acc += prob6 * prob8 * prob10 * prob12 * pPerfect[stateIndex(keep6, keep8, keep10, keep12)];
          }
        }
        pPerfect[stateIndex(d6, d8, d10, d12)] = acc;
      }
    }
}

int main(int argc, char** argv){
    long N = (argc > 1) ? atol(argv[1]) : 50000000;
    solveV();
    solveP1();

    double p = pPerfect[stateIndex(12, 1, 1, 1)];
    printf("EXACT P(perfect | optimal play) = %.6e  = 1 in %.0f  (%.4f%%)\n", p, 1.0/p, 100*p);

    // MC validation: play optimal, count score == 0.
    std::mt19937_64 rng(0x50FA);
    long zeros = 0;
    for(long t = 0; t < N; ++t){
        int d6 = 12, d8 = 1, d10 = 1, d12 = 1;
        int sc = 0;
        while(d6 + d8 + d10 + d12 > 0){
            int faces[12];
            for(int j = 0; j < d6; ++j) faces[j] = rng() % 6;
            std::sort(faces, faces + d6, std::greater<int>());
            double topPenSum6[13];
            topPenSum6[0] = 0;
            for(int k = 0; k < d6; ++k) topPenSum6[k+1] = topPenSum6[k] + faces[k];

            int pen8  = d8  ? rng() % 8  : 0;
            int pen10 = d10 ? rng() % 10 : 0;
            int pen12 = d12 ? rng() % 12 : 0;

            int keep6, keep8, keep10, keep12;
            optAction(d6, d8, d10, d12, topPenSum6, pen8, pen10, pen12,
                      keep6, keep8, keep10, keep12);

            int totalPen = (int)topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
            int keptPen  = (int)topPenSum6[keep6] + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
            sc += totalPen - keptPen;
            d6 = keep6;
            d8 = keep8;
            d10 = keep10;
            d12 = keep12;
        }
        if(sc == 0) zeros++;
    }
    printf("MC P(perfect | optimal) = %.6e  (%ld/%ld)\n", (double)zeros/N, zeros, N);
    return 0;
}
