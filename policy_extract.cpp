// Extract the PROVABLY OPTIMAL policy for bitches and test separability.
//
// 1. DP over 104 states -> optimal value V[S] (lower = better).
// 2. For each state S, enumerate every roll exactly. For each roll the optimal
//    action keeps a set K ⊊ S maximizing  (sum of kept penalties) - V(K).
//    Within a type you keep the highest-penalty dice, so an action is a per-type
//    kept-count. We track, over all optimal actions of each roll, whether a die
//    of (type t, penalty p) is *forced banked* (banked in every optimal action),
//    *forced kept*, or free. A per-die threshold "bank iff penalty <= T_t" is
//    OPTIMAL iff no penalty p is forced-banked in one roll and forced-kept in
//    another. We derive T_t and flag non-separable states.
// 3. Verify: replay the extracted thresholds (forced = optimal single removal)
//    with optimal continuation V; if the one-step value equals V[S] for every
//    state, the threshold policy is provably globally optimal.
//
// Build: c++ -O3 -mcpu=native -o policy_extract policy_extract.cpp

#include <cstdio>
#include <cmath>

// Pack a state (d6 in 0..12; d8,d10,d12 in {0,1}) into a unique index 0..103.
static inline int stateIndex(int d6, int d8, int d10, int d12){
    return ((d6 * 2 + d8) * 2 + d10) * 2 + d12;
}

static double V[104];           // V[state] = optimal expected remaining score
static double factorial[13];    // factorial[n] = n!
static double pow6[13];         // pow6[n] = 6^n
static const int SZ[4] = {6, 8, 10, 12};   // die sizes by type index

// penalty-cutoff per state per type (bank iff penalty <= Tt); -1 = never bank
static int Tt[104][4];
static bool nonsep[104];
static double oneStepGap[104];

int main(){
    factorial[0] = 1;
    for(int i = 1; i < 13; ++i) factorial[i] = factorial[i - 1] * i;
    for(int n = 0; n < 13; ++n) pow6[n] = pow(6.0, n);

    // ---------- pass A: optimal DP ----------
    V[stateIndex(0, 0, 0, 0)] = 0;
    for(int totalDice = 1; totalDice <= 15; ++totalDice)
     for(int d6 = 0; d6 <= 12; ++d6)
     for(int d8 = 0; d8 <= 1; ++d8)
     for(int d10 = 0; d10 <= 1; ++d10)
     for(int d12 = 0; d12 <= 1; ++d12){
      if(d6 + d8 + d10 + d12 != totalDice) continue;
      double expectedScore = 0;

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
            int k = 0;
            int faceCount[6] = {face1, face2, face3, face4, face5, face6};
            for(int pen = 5, c = 0; pen >= 0; --pen, ++c)
                for(int j = 0; j < faceCount[c]; ++j){
                    topPenSum6[k + 1] = topPenSum6[k] + pen;
                    ++k;
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
          double totalPen = totalPen6 + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

          double bestKeepValue = -1e300;
          for(int keep6  = 0; keep6  <= d6;  ++keep6)
          for(int keep8  = 0; keep8  <= d8;  ++keep8)
          for(int keep10 = 0; keep10 <= d10; ++keep10)
          for(int keep12 = 0; keep12 <= d12; ++keep12){
            if(keep6 == d6 && keep8 == d8 && keep10 == d10 && keep12 == d12) continue;
            double value = topPenSum6[keep6] + keep8Pen[keep8] + keep10Pen[keep10]
                + keep12Pen[keep12] - V[stateIndex(keep6, keep8, keep10, keep12)];
            if(value > bestKeepValue) bestKeepValue = value;
          }
          expectedScore += prob6 * prob8 * prob10 * prob12 * (totalPen - bestKeepValue);
        }
      }
      V[stateIndex(d6, d8, d10, d12)] = expectedScore;
     }
    printf("OPTIMAL expected score V(12,1,1,1) = %.6f\n\n", V[stateIndex(12, 1, 1, 1)]);

    // ---------- pass B: per-state separability + threshold extraction ----------
    for(int s = 0; s < 104; ++s){
        nonsep[s] = false;
        for(int t = 0; t < 4; ++t) Tt[s][t] = -1;
    }

    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
      int totalDice = d6 + d8 + d10 + d12;
      if(totalDice == 0) continue;
      int S = stateIndex(d6, d8, d10, d12);
      // canBank[t][p]/canKeep[t][p]: across VOLUNTARY rolls, is a (type t, penalty
      // p) die banked in some optimal action / kept in some optimal action? If a
      // penalty is BOTH banked-somewhere and kept-somewhere, no fixed per-die
      // threshold can reproduce the optimum -> non-separable.
      bool canBank[4][12] = {}, canKeep[4][12] = {};

      for(int face1 = 0; face1 <= d6; ++face1)
      for(int face2 = 0; face2 <= d6 - face1; ++face2)
      for(int face3 = 0; face3 <= d6 - face1 - face2; ++face3)
      for(int face4 = 0; face4 <= d6 - face1 - face2 - face3; ++face4)
      for(int face5 = 0; face5 <= d6 - face1 - face2 - face3 - face4; ++face5){
        int face6 = d6 - face1 - face2 - face3 - face4 - face5;
        int faceCount[6] = {face1, face2, face3, face4, face5, face6};   // counts of pen 5,4,3,2,1,0

        // topPenSum6[k] = sum of the k largest d6 penalties this roll.
        double topPenSum6[13];
        topPenSum6[0] = 0;
        {
            int k = 0;
            for(int pen = 5, c = 0; pen >= 0; --pen, ++c)
                for(int j = 0; j < faceCount[c]; ++j){
                    topPenSum6[k + 1] = topPenSum6[k] + pen;
                    ++k;
                }
        }
        for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
        for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
        for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
          double keep8Pen[2]  = {0, (double)pen8};
          double keep10Pen[2] = {0, (double)pen10};
          double keep12Pen[2] = {0, (double)pen12};
          // find optimal value M over all legal kept-compositions
          double M = -1e300;
          for(int keep6  = 0; keep6  <= d6;  ++keep6)
          for(int keep8  = 0; keep8  <= d8;  ++keep8)
          for(int keep10 = 0; keep10 <= d10; ++keep10)
          for(int keep12 = 0; keep12 <= d12; ++keep12){
            if(keep6 == d6 && keep8 == d8 && keep10 == d10 && keep12 == d12) continue;
            double value = topPenSum6[keep6] + keep8Pen[keep8] + keep10Pen[keep10]
                + keep12Pen[keep12] - V[stateIndex(keep6, keep8, keep10, keep12)];
            if(value > M) M = value;
          }
          // If keeping EVERYTHING would be best (but is illegal), this roll is a
          // FORCED bank — banking here is involuntary, so it tells us nothing
          // about the voluntary per-die threshold. Skip it for extraction.
          double totalPen = topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
          double gFull = totalPen - V[S];
          if(gFull >= M - 1e-9) continue;
          // min/max kept-count per type over all optimal actions
          int kmn[4] = {99, 99, 99, 99}, kmx[4] = {-1, -1, -1, -1};
          for(int keep6  = 0; keep6  <= d6;  ++keep6)
          for(int keep8  = 0; keep8  <= d8;  ++keep8)
          for(int keep10 = 0; keep10 <= d10; ++keep10)
          for(int keep12 = 0; keep12 <= d12; ++keep12){
            if(keep6 == d6 && keep8 == d8 && keep10 == d10 && keep12 == d12) continue;
            double value = topPenSum6[keep6] + keep8Pen[keep8] + keep10Pen[keep10]
                + keep12Pen[keep12] - V[stateIndex(keep6, keep8, keep10, keep12)];
            if(value > M - 1e-9){
              int kk[4] = {keep6, keep8, keep10, keep12};
              for(int t = 0; t < 4; ++t){
                if(kk[t] < kmn[t]) kmn[t] = kk[t];
                if(kk[t] > kmx[t]) kmx[t] = kk[t];
              }
            }
          }
          // d6: for each penalty value p present, hi = #d6 with pen>p, eq = #d6 with pen==p
          if(d6){
            for(int p = 0; p <= 5; ++p){
              int eq = faceCount[5 - p];
              if(!eq) continue;
              int hi = 0;
              for(int pp = p + 1; pp <= 5; ++pp) hi += faceCount[5 - pp];
              if(kmn[0] < hi + eq) canBank[0][p] = true;   // some optimal banks a pen-p d6
              if(kmx[0] > hi)      canKeep[0][p] = true;   // some optimal keeps a pen-p d6
            }
          }
          // big dice (single die each): hi=0, eq=1
          int pbig[4]    = {0, pen8, pen10, pen12};
          int present[4] = {d6, d8, d10, d12};
          for(int t = 1; t < 4; ++t){
            if(!present[t]) continue;
            int p = pbig[t];
            if(kmn[t] < 1) canBank[t][p] = true;
            if(kmx[t] > 0) canKeep[t][p] = true;
          }
        }
      }
      // classify each present type
      int present[4] = {d6, d8, d10, d12};
      for(int t = 0; t < 4; ++t){
        if(!present[t]) continue;
        int sz = SZ[t];
        bool bad = false;
        for(int p = 0; p < sz; ++p) if(canBank[t][p] && canKeep[t][p]) bad = true;   // banked & kept somewhere
        // separable threshold: bank iff penalty < (lowest penalty ever kept).
        int minkeep = 999;
        for(int p = 0; p < sz; ++p) if(canKeep[t][p] && p < minkeep) minkeep = p;
        Tt[S][t] = (minkeep == 999) ? sz - 1 : minkeep - 1;   // never kept => bank all
        if(bad) nonsep[S] = true;
      }
    }

    // ---------- pass C: verify extracted thresholds are one-step optimal ----------
    int worst = -1;
    double worstgap = 0;
    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
      int totalDice = d6 + d8 + d10 + d12;
      if(totalDice == 0) continue;
      int S = stateIndex(d6, d8, d10, d12);
      int T6 = Tt[S][0], T8 = Tt[S][1], T10 = Tt[S][2], T12 = Tt[S][3];
      double E = 0;
      for(int face1 = 0; face1 <= d6; ++face1)
      for(int face2 = 0; face2 <= d6 - face1; ++face2)
      for(int face3 = 0; face3 <= d6 - face1 - face2; ++face3)
      for(int face4 = 0; face4 <= d6 - face1 - face2 - face3; ++face4)
      for(int face5 = 0; face5 <= d6 - face1 - face2 - face3 - face4; ++face5){
        int face6 = d6 - face1 - face2 - face3 - face4 - face5;
        int faceCount[6] = {face1, face2, face3, face4, face5, face6};
        double prob6 = (factorial[d6] /
            (factorial[face1] * factorial[face2] * factorial[face3] *
             factorial[face4] * factorial[face5] * factorial[face6]))
            / pow6[d6];
        // d6 kept (pen>T6) and banked penalty
        int kept6 = 0;
        double keptPen6 = 0;
        int lowPen6 = 99;
        for(int c = 0, pen = 5; c < 6; ++c, --pen){
            if(faceCount[c] > 0 && pen < lowPen6) lowPen6 = pen;
            if(pen > T6){
                kept6 += faceCount[c];
                keptPen6 += (double)pen * faceCount[c];
            }
        }
        double totalPen6 = 0;
        for(int c = 0, pen = 5; c < 6; ++c, --pen) totalPen6 += (double)pen * faceCount[c];
        double prob8  = d8  ? 1.0 / 8  : 1;
        double prob10 = d10 ? 1.0 / 10 : 1;
        double prob12 = d12 ? 1.0 / 12 : 1;
        for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
        for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
        for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
          int keep8  = (d8  && pen8  > T8);
          int keep10 = (d10 && pen10 > T10);
          int keep12 = (d12 && pen12 > T12);
          int K6 = kept6, K8 = keep8, K10 = keep10, K12 = keep12;
          double keptPen = keptPen6 + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
          double totalPen = totalPen6 + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
          double banked = totalPen - keptPen;
          int u6 = K6, u8 = K8, u10 = K10, u12 = K12;
          if(K6 == d6 && K8 == d8 && K10 == d10 && K12 == d12){   // forced: optimal single removal
            double bestv = 1e300;
            if(d6){
                double v = lowPen6 + V[stateIndex(d6 - 1, d8, d10, d12)];
                if(v < bestv){ bestv = v; u6 = d6 - 1; u8 = d8; u10 = d10; u12 = d12; }
            }
            if(d8){
                double v = pen8 + V[stateIndex(d6, d8 - 1, d10, d12)];
                if(v < bestv){ bestv = v; u6 = d6; u8 = d8 - 1; u10 = d10; u12 = d12; }
            }
            if(d10){
                double v = pen10 + V[stateIndex(d6, d8, d10 - 1, d12)];
                if(v < bestv){ bestv = v; u6 = d6; u8 = d8; u10 = d10 - 1; u12 = d12; }
            }
            if(d12){
                double v = pen12 + V[stateIndex(d6, d8, d10, d12 - 1)];
                if(v < bestv){ bestv = v; u6 = d6; u8 = d8; u10 = d10; u12 = d12 - 1; }
            }
            E += prob6 * prob8 * prob10 * prob12 * bestv;
            continue;
          }
          E += prob6 * prob8 * prob10 * prob12 * (banked + V[stateIndex(u6, u8, u10, u12)]);
        }
      }
      double gap = E - V[S];
      oneStepGap[S] = gap;
      if(gap > worstgap){ worstgap = gap; worst = S; }
    }

    // ---------- pass D: full-game value of the extracted threshold policy ----------
    // (recursive: forced bank = optimal single removal, policy's own continuation)
    static double Vpol[104];
    Vpol[stateIndex(0, 0, 0, 0)] = 0;
    for(int totalDice = 1; totalDice <= 15; ++totalDice)
     for(int d6 = 0; d6 <= 12; ++d6)
     for(int d8 = 0; d8 <= 1; ++d8)
     for(int d10 = 0; d10 <= 1; ++d10)
     for(int d12 = 0; d12 <= 1; ++d12){
      if(d6 + d8 + d10 + d12 != totalDice) continue;
      int S = stateIndex(d6, d8, d10, d12);
      int T6 = Tt[S][0], T8 = Tt[S][1], T10 = Tt[S][2], T12 = Tt[S][3];
      double E = 0;
      for(int face1 = 0; face1 <= d6; ++face1)
      for(int face2 = 0; face2 <= d6 - face1; ++face2)
      for(int face3 = 0; face3 <= d6 - face1 - face2; ++face3)
      for(int face4 = 0; face4 <= d6 - face1 - face2 - face3; ++face4)
      for(int face5 = 0; face5 <= d6 - face1 - face2 - face3 - face4; ++face5){
        int face6 = d6 - face1 - face2 - face3 - face4 - face5;
        int faceCount[6] = {face1, face2, face3, face4, face5, face6};
        double prob6 = (factorial[d6] /
            (factorial[face1] * factorial[face2] * factorial[face3] *
             factorial[face4] * factorial[face5] * factorial[face6]))
            / pow6[d6];
        int kept6 = 0;
        double keptPen6 = 0;
        int lowPen6 = 99;
        for(int c = 0, pen = 5; c < 6; ++c, --pen){
            if(faceCount[c] > 0 && pen < lowPen6) lowPen6 = pen;
            if(pen > T6){
                kept6 += faceCount[c];
                keptPen6 += (double)pen * faceCount[c];
            }
        }
        double totalPen6 = 0;
        for(int c = 0, pen = 5; c < 6; ++c, --pen) totalPen6 += (double)pen * faceCount[c];
        double prob8  = d8  ? 1.0 / 8  : 1;
        double prob10 = d10 ? 1.0 / 10 : 1;
        double prob12 = d12 ? 1.0 / 12 : 1;
        for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
        for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
        for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
          int keep8  = (d8  && pen8  > T8);
          int keep10 = (d10 && pen10 > T10);
          int keep12 = (d12 && pen12 > T12);
          int K6 = kept6, K8 = keep8, K10 = keep10, K12 = keep12;
          double keptPen = keptPen6 + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
          double totalPen = totalPen6 + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
          double banked = totalPen - keptPen;
          int u6 = K6, u8 = K8, u10 = K10, u12 = K12;
          if(K6 == d6 && K8 == d8 && K10 == d10 && K12 == d12){
            double bv = 1e300;
            if(d6){
                double v = lowPen6 + Vpol[stateIndex(d6 - 1, d8, d10, d12)];
                if(v < bv){ bv = v; u6 = d6 - 1; u8 = d8; u10 = d10; u12 = d12; }
            }
            if(d8){
                double v = pen8 + Vpol[stateIndex(d6, d8 - 1, d10, d12)];
                if(v < bv){ bv = v; u6 = d6; u8 = d8 - 1; u10 = d10; u12 = d12; }
            }
            if(d10){
                double v = pen10 + Vpol[stateIndex(d6, d8, d10 - 1, d12)];
                if(v < bv){ bv = v; u6 = d6; u8 = d8; u10 = d10 - 1; u12 = d12; }
            }
            if(d12){
                double v = pen12 + Vpol[stateIndex(d6, d8, d10, d12 - 1)];
                if(v < bv){ bv = v; u6 = d6; u8 = d8; u10 = d10; u12 = d12 - 1; }
            }
            E += prob6 * prob8 * prob10 * prob12 * bv;
            continue;
          }
          E += prob6 * prob8 * prob10 * prob12 * (banked + Vpol[stateIndex(u6, u8, u10, u12)]);
        }
      }
      Vpol[S] = E;
     }

    // ---------- report ----------
    int nNon = 0;
    for(int s = 0; s < 104; ++s) if(nonsep[s]) ++nNon;
    printf("Separability: %d / 104 states have a NON-separable optimal action.\n", nNon);
    printf("Verification: extracted-threshold policy worst one-step gap vs optimal = %.2e\n", worstgap);
    printf("Full-game value of the extracted cutoff policy = %.4f  (optimal = %.4f, loss %.4f)\n",
           Vpol[stateIndex(12, 1, 1, 1)], V[stateIndex(12, 1, 1, 1)],
           Vpol[stateIndex(12, 1, 1, 1)] - V[stateIndex(12, 1, 1, 1)]);
    {
      int d6 = worst / 8, r = worst % 8;
      (void)r;
      printf("worst one-step state idx=%d, gap=%.4f\n", worst, worstgap);
      (void)d6;
    }
    printf("\n");

    printf("Optimal bank cutoffs  (bank a die iff face >= cutoff;  '-' = type absent):\n");
    printf("state(a6,a8,a10,a12) | d6  d8  d10 d12 | sep? | V(S)\n");
    for(int d12 = 0; d12 <= 1; ++d12)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d6 = 12; d6 >= 0; --d6){
      int totalDice = d6 + d8 + d10 + d12;
      if(totalDice == 0) continue;
      int S = stateIndex(d6, d8, d10, d12);
      int present[4] = {d6, d8, d10, d12};
      char buf[4][6];
      for(int t = 0; t < 4; ++t){
        if(!present[t]){
            snprintf(buf[t], 6, " - ");
        } else {
            int cut = SZ[t] - Tt[S][t];
            if(Tt[S][t] < 0) cut = SZ[t] + 1;   // never bank
            snprintf(buf[t], 6, "%2d ", cut);
        }
      }
      printf("    (%2d,%d,%d,%d)        %s %s %s %s   %s    %.3f\n",
             d6, d8, d10, d12, buf[0], buf[1], buf[2], buf[3], nonsep[S] ? "NO " : "yes", V[S]);
    }
    return 0;
}
