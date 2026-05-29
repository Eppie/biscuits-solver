// STRICT separability test for the optimal policy (tie-immune).
// For each state, type t and penalty p, look across all rolls at:
//   vKeep = best achievable value while KEEPING a pen-p die of type t
//   vBank = best achievable value while BANKING a pen-p die
// (value = sum kept penalties - V(kept)). If in some roll vBank > vKeep
// (strictly want to bank) and in another vKeep > vBank (strictly want to keep)
// for the SAME (t,p), then no fixed per-die cutoff is optimal -> non-separable.
// Ties (vBank==vKeep, i.e. either choice is optimal) never trigger a flag.
//
// Build: c++ -O3 -mcpu=native -o sep_strict sep_strict.cpp

#include <cstdio>
#include "biscuits.h"

static double V[NUM_STATES];    // V[state] = optimal expected remaining score
static const int SZ[4] = {6, 8, 10, 12};   // die sizes by type index

int main(){
    // ---------- optimal value function V ----------
    solveV(V);
    printf("OPTIMAL V(12,1,1,1) = %.6f\n\n", V[stateIndex(12, 1, 1, 1)]);

    // ---------- strict separability + cutoff per state ----------
    int nNon = 0;
    int cutFace[104][4];
    bool sep[104];
    for(int s = 0; s < 104; ++s){
        sep[s] = true;
        for(int t = 0; t < 4; ++t) cutFace[s][t] = 99;
    }

    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
      if(d6 + d8 + d10 + d12 == 0) continue;
      int S = stateIndex(d6, d8, d10, d12);
      // strictBank[t][p] / strictKeep[t][p]: in some roll do we STRICTLY prefer
      // to bank / keep a (type t, penalty p) die under the optimal value?
      bool strictBank[4][12] = {}, strictKeep[4][12] = {};
      const double EPS = 1e-7;

      for(int face1 = 0; face1 <= d6; ++face1)
      for(int face2 = 0; face2 <= d6 - face1; ++face2)
      for(int face3 = 0; face3 <= d6 - face1 - face2; ++face3)
      for(int face4 = 0; face4 <= d6 - face1 - face2 - face3; ++face4)
      for(int face5 = 0; face5 <= d6 - face1 - face2 - face3 - face4; ++face5){
        int face6 = d6 - face1 - face2 - face3 - face4 - face5;
        int faceCount[6] = {face1, face2, face3, face4, face5, face6};

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
          // bestByK[t][k] = max value over kept-compositions with kept-count of type t == k
          double bestByK[4][13];
          for(int t = 0; t < 4; ++t)
              for(int k = 0; k < 13; ++k) bestByK[t][k] = -1e300;
          for(int keep6  = 0; keep6  <= d6;  ++keep6)
          for(int keep8  = 0; keep8  <= d8;  ++keep8)
          for(int keep10 = 0; keep10 <= d10; ++keep10)
          for(int keep12 = 0; keep12 <= d12; ++keep12){
            if(keep6 == d6 && keep8 == d8 && keep10 == d10 && keep12 == d12) continue;
            double value = topPenSum6[keep6] + keep8Pen[keep8] + keep10Pen[keep10]
                + keep12Pen[keep12] - V[stateIndex(keep6, keep8, keep10, keep12)];
            if(value > bestByK[0][keep6])  bestByK[0][keep6]  = value;
            if(value > bestByK[1][keep8])  bestByK[1][keep8]  = value;
            if(value > bestByK[2][keep10]) bestByK[2][keep10] = value;
            if(value > bestByK[3][keep12]) bestByK[3][keep12] = value;
          }

          // d6: for each penalty p present, hi = #pen>p, eq = #pen==p
          if(d6) for(int p = 0; p <= 5; ++p){
            int eq = faceCount[5 - p];
            if(!eq) continue;
            int hi = 0;
            for(int pp = p + 1; pp <= 5; ++pp) hi += faceCount[5 - pp];
            double vKeep = -1e300;   // best value among actions that keep a pen-p d6
            for(int k = hi + 1; k <= d6; ++k) if(bestByK[0][k] > vKeep) vKeep = bestByK[0][k];
            double vBank = -1e300;   // best value among actions that bank a pen-p d6
            for(int k = 0; k <= hi + eq - 1; ++k) if(bestByK[0][k] > vBank) vBank = bestByK[0][k];
            if(vBank > vKeep + EPS) strictBank[0][p] = true;
            if(vKeep > vBank + EPS) strictKeep[0][p] = true;
          }

          // big dice (single die each): hi = 0, eq = 1
          int pbig[4]    = {0, pen8, pen10, pen12};
          int present[4] = {d6, d8, d10, d12};
          for(int t = 1; t < 4; ++t){
            if(!present[t]) continue;
            int p = pbig[t];
            double vKeep = bestByK[t][1];   // keep the die
            double vBank = bestByK[t][0];   // bank the die
            if(vBank > vKeep + EPS) strictBank[t][p] = true;
            if(vKeep > vBank + EPS) strictKeep[t][p] = true;
          }
        }
      }

      // classify each present type
      int present[4] = {d6, d8, d10, d12};
      for(int t = 0; t < 4; ++t){
        if(!present[t]) continue;
        int sz = SZ[t];
        bool bad = false;
        for(int p = 0; p < sz; ++p) if(strictBank[t][p] && strictKeep[t][p]) bad = true;
        // cutoff: bank iff penalty <= T; T = highest p that is ever strict-bank.
        int T = -1;
        for(int p = 0; p < sz; ++p) if(strictBank[t][p]) T = p;
        cutFace[S][t] = sz - T;          // bank iff face >= this (T=-1 -> sz+1 = never)
        if(T < 0) cutFace[S][t] = sz + 1;
        if(bad) sep[S] = false;
      }
      if(!sep[S]) ++nNon;
    }

    // ---------- report ----------
    printf("STRICT separability: %d / 104 states have a genuinely NON-separable optimal action.\n\n", nNon);
    printf("Optimal bank cutoffs (bank iff face >= X; '-' absent; '*' = type non-separable here):\n");
    printf("state(a6,a8,a10,a12) | d6  d8  d10 d12 | V(S)\n");
    for(int d12 = 0; d12 <= 1; ++d12)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d6 = 12; d6 >= 0; --d6){
      if(d6 + d8 + d10 + d12 == 0) continue;
      int S = stateIndex(d6, d8, d10, d12);
      int present[4] = {d6, d8, d10, d12};
      char b[4][6];
      for(int t = 0; t < 4; ++t){
        if(!present[t]){
            snprintf(b[t], 6, " - ");
        } else {
            int c = cutFace[S][t];
            if(c > SZ[t]) snprintf(b[t], 6, " x ");   // never voluntarily bank
            else          snprintf(b[t], 6, "%2d ", c);
        }
      }
      printf("    (%2d,%d,%d,%d)        %s %s %s %s  %s %.3f\n", d6, d8, d10, d12, b[0], b[1], b[2], b[3],
             sep[S] ? "  " : "NS", V[S]);
    }
    return 0;
}
