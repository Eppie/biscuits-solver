// "Decision card" variant: precompute per-big-config rank thresholds
//   keepThreshold[b][j] = V(keep j d6 | big-config b) - V(keep j-1 d6 | b),
// then each roll keep the j-th-worst d6 while its penalty > keepThreshold[b][j]
// (stop at the first that isn't), choosing the big-config by resulting value.
// Tests whether this is (a) faster than the argmax and (b) still optimal.
//
// A game state is the multiset of dice still in hand:
//   d6                 : number of six-sided dice left, 0..12
//   d8, d10, d12       : whether that single big die is still in hand, 0 or 1
// That is 13 * 2 * 2 * 2 = 104 states. The big-config index b = d8*4 + d10*2 + d12
// (0..7), and stateIndex(d6,d8,d10,d12) == d6 * 8 + b.
//
// Build: c++ -O3 -mcpu=native -o opt_card opt_card.cpp

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>

// Pack a state (d6 in 0..12; d8,d10,d12 in {0,1}) into a unique index 0..103.
static inline int stateIndex(int d6, int d8, int d10, int d12){
    return ((d6 * 2 + d8) * 2 + d10) * 2 + d12;
}

static double V[104];           // V[state] = optimal expected remaining score
static double factorial[13];    // factorial[n] = n!
static double pow6[13];         // pow6[n] = 6^n
// keepThreshold[b][j] = marginal value of keeping the j-th d6 given big-config b.
static float keepThreshold[8][13];

// Solve the value function V exactly, then derive the decision-card thresholds.
static void solve(){
    factorial[0] = 1;
    for(int i = 1; i < 13; ++i) factorial[i] = factorial[i-1] * i;
    for(int n = 0; n < 13; ++n) pow6[n] = pow(6.0, n);

    V[0] = 0;

    for(int totalDice = 1; totalDice <= 15; ++totalDice){
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
              double keep8Pen[2]  = {0, (double)pen8};
              double keep10Pen[2] = {0, (double)pen10};
              double keep12Pen[2] = {0, (double)pen12};

              double totalPen = topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

              double bestKeepValue = -1e300;
              for(int keep6  = 0; keep6  <= d6;  ++keep6)
              for(int keep8  = 0; keep8  <= d8;  ++keep8)
              for(int keep10 = 0; keep10 <= d10; ++keep10)
              for(int keep12 = 0; keep12 <= d12; ++keep12){
                  if(keep6 == d6 && keep8 == d8 && keep10 == d10 && keep12 == d12) continue;
                  double value = topPenSum6[keep6] + keep8Pen[keep8] + keep10Pen[keep10] + keep12Pen[keep12]
                      - V[stateIndex(keep6, keep8, keep10, keep12)];
                  if(value > bestKeepValue) bestKeepValue = value;
              }

              expectedScore += prob6 * prob8 * prob10 * prob12 * (totalPen - bestKeepValue);
          }
        }
        V[stateIndex(d6, d8, d10, d12)] = expectedScore;
      }
    }

    // Derive thresholds: V[j*8+b] == V(keep j d6 | big-config b).
    for(int b = 0; b < 8; ++b)
        for(int j = 1; j <= 12; ++j)
            keepThreshold[b][j] = (float)(V[j*8 + b] - V[(j-1)*8 + b]);
}

// xorshift128+ RNG (fixed seeds for reproducible Monte Carlo).
static uint64_t rngState0 = 0x243F6A8885A308D3ULL;
static uint64_t rngState1 = 0x13198A2E03707344ULL;

static inline uint64_t xorshift(){
    uint64_t x = rngState0, y = rngState1;
    rngState0 = y;
    x ^= x << 23;
    rngState1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rngState1 + y;
}

// Roll a fair die with s faces, returning 0..s-1.
static inline int rollDie(int s){
    return (int)(((xorshift() >> 32) * (uint64_t)s) >> 32);
}

int main(int argc, char** argv){
    long N = (argc > 1) ? atol(argv[1]) : 20000000;
    solve();
    printf("optimal V(12,1,1,1)=%.6f\n", V[stateIndex(12, 1, 1, 1)]);

    double sum = 0, rolls = 0;
    auto t0 = std::chrono::steady_clock::now();
    for(long t = 0; t < N; ++t){
        int d6 = 12, d8 = 1, d10 = 1, d12 = 1;
        int sc = 0;
        while(d6 + d8 + d10 + d12 > 0){
            rolls++;
            int h[6] = {0, 0, 0, 0, 0, 0};
            for(int j = 0; j < d6; ++j) h[rollDie(6)]++;
            int pe[13];
            {
                int k = 0;
                for(int penalty = 5; penalty >= 0; --penalty)
                    for(int c = 0; c < h[penalty]; ++c) pe[++k] = penalty;
            }   // pe[1..d6] = d6 penalties in descending order
            int topPenSum6[13];
            topPenSum6[0] = 0;
            for(int k = 1; k <= d6; ++k) topPenSum6[k] = topPenSum6[k-1] + pe[k];

            int pen8  = d8  ? rollDie(8)  : 0;
            int pen10 = d10 ? rollDie(10) : 0;
            int pen12 = d12 ? rollDie(12) : 0;
            int totalPen = topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);

            // CARD: for each big-config, keep6 = keep-while penalty > threshold
            // (stop at first that isn't), then pick the best big-config by value.
            float best = -1e30f;
            int keep6 = 0, keep8 = 0, keep10 = 0, keep12 = 0;
            for(int k8 = 0; k8 <= d8; ++k8)
            for(int k10 = 0; k10 <= d10; ++k10)
            for(int k12 = 0; k12 <= d12; ++k12){
                int b = k8 * 4 + k10 * 2 + k12;
                float bigKept = (float)((k8 ? pen8 : 0) + (k10 ? pen10 : 0) + (k12 ? pen12 : 0));
                int k6 = 0;
                while(k6 < d6 && (float)pe[k6+1] > keepThreshold[b][k6+1]) k6++;   // threshold scan
                int bigFull = (k8 == d8 && k10 == d10 && k12 == d12);
                if(bigFull && k6 == d6) k6--;   // can't keep everything
                if(k6 < 0) k6 = 0;
                float v = (float)topPenSum6[k6] + bigKept - (float)V[k6*8 + b];
                if(v > best){
                    best = v;
                    keep6 = k6;
                    keep8 = k8;
                    keep10 = k10;
                    keep12 = k12;
                }
            }
            int keptPen = topPenSum6[keep6] + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
            sc += totalPen - keptPen;
            d6 = keep6;
            d8 = keep8;
            d10 = keep10;
            d12 = keep12;
        }
        sum += sc;
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mean = sum / N;
    printf("CARD: %.3f M games/s, %.1f M moves/s  | mean=%.5f (optimal=%.5f, gap=%+.5f)\n",
      N/secs/1e6, rolls/secs/1e6, mean, V[stateIndex(12, 1, 1, 1)], mean - V[stateIndex(12, 1, 1, 1)]);
    return 0;
}
