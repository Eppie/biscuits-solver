// bitches (a dice game) — COMPETITIVE N-player strategy ("lowest score wins").
//
// Everything in exact_dp.cpp minimizes the EXPECTED score (single agent). The
// actual game is won by having the LOWEST score at the table, which is a
// different objective: you don't care how low you go, only about beating the
// field. That rewards variance management (gamble when behind, play safe when
// ahead) and, for symmetric players, becomes a game-theoretic equilibrium.
//
// This program implements the full plan from TODO.md item 1:
//
//  (A) SCORE-DISTRIBUTION DP. Extend the value DP to carry the full distribution
//      of final scores per state (not just the mean). For the expected-optimal
//      policy this gives mean/variance/percentiles/P(perfect) for free.
//
//  (B) WIN-VALUE W(s). Given (N-1) i.i.d. opponents each with score distribution
//      D, a player who finishes at score s earns expected win-share
//        W(s) = ( P(opp>=s)^N - P(opp>s)^N ) / ( N * P(opp=s) )      (=ge^(N-1) if P(opp=s)=0)
//      i.e. outright win if all opponents are higher, split N-ways on ties.
//      This is the terminal payoff (higher is better; W is non-increasing in s).
//
//  (C) WIN-PROBABILITY DP U(S,g). State = (hand-state S, points banked so far g).
//        U(S,g) = E_roll [ max over kept K<S  U(K, g + banked(roll,K)) ]
//        U(empty,g) = W(g)
//      U(full,0) is the player's expected win-share. The maximizing action
//      depends on g — that's the variance management the mean-optimal policy lacks.
//
//  (D) BEST RESPONSE to "everyone plays expected-optimal": field D = expected-
//      optimal score distribution. a(N)=U(full,0) > 1/N measures the exploitation.
//
//  (E) SYMMETRIC EQUILIBRIUM via fictitious play: iterate D <- (score dist of the
//      best response to (N-1) opponents drawn from D) until D is a fixed point.
//
//  (F) VALIDATION: multi-player Monte-Carlo tournament confirms the DP win-shares.
//
// Score support: a d6 face f has penalty 6-f (<=5), so 12 d6 <= 60; d8<=7, d10<=9,
// d12<=11. Max total score = 87. Bound g per state by reachability (see maxBankedOf).
//
// Build: c++ -O3 -mcpu=native -std=c++17 -pthread -o competitive competitive.cpp
//        (use -march=native instead of -mcpu=native on x86; or just `make competitive`)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>

#include "bitches.h"

static const int MAXS = 87;     // maximum possible final score
static const int SZ   = 88;     // score-index size (0..87)

static double V[NUM_STATES];    // V[state] = expected-optimal value function (mean remaining score)
static double factorial[13];    // factorial[n] = n!
static double pow6[13];         // pow6[n] = 6^n
static int    maxRemainOf[104]; // max remaining penalty reachable from a state
static int    maxBankedOf[104]; // max banked-so-far g that can reach this state = 87 - maxRemain(S)
static int    numThreads = 1;   // worker threads for the (S,g) DP passes

// Maximum penalty still attainable from a hand: 5 per d6, plus the big-die maxima.
static inline int maxRemainingPen(int d6, int d8, int d10, int d12){
    return 5 * d6 + 7 * d8 + 9 * d10 + 11 * d12;
}

static void initTables(){
    factorial[0] = 1;
    for(int i = 1; i < 13; ++i){
        factorial[i] = factorial[i-1] * i;
    }
    for(int n = 0; n < 13; ++n){
        pow6[n] = pow(6.0, n);
    }
    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
        int m  = maxRemainingPen(d6, d8, d10, d12);
        int id = stateIndex(d6, d8, d10, d12);
        maxRemainOf[id]  = m;
        maxBankedOf[id]  = MAXS - m;
    }
}

// =====================================================================
// (A) Score distribution of the EXPECTED-OPTIMAL policy.
//     Dopt[S][v] = P(additional penalty from S = v) under optimal play.
// =====================================================================
static std::vector<double> Dopt[104];

static void scoreDistOptimal(){
    for(int s = 0; s < 104; ++s){
        Dopt[s].assign(SZ, 0.0);
    }
    Dopt[stateIndex(0, 0, 0, 0)][0] = 1.0;
    for(int totalDice = 1; totalDice <= 15; ++totalDice)
    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice){
            continue;
        }
        int sid = stateIndex(d6, d8, d10, d12);
        std::vector<double>& D = Dopt[sid];   // distribution being filled for this state
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
            double prob8  = d8  ? 1.0 / 8  : 1;
            double prob10 = d10 ? 1.0 / 10 : 1;
            double prob12 = d12 ? 1.0 / 12 : 1;
            for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
            for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
            for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
                int keep6, keep8, keep10, keep12;
                optimalKeep(V, d6, d8, d10, d12, topPenSum6, pen8, pen10, pen12,
                      keep6, keep8, keep10, keep12);
                int totalPen = (int)topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
                int keptPen  = (int)topPenSum6[keep6] + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
                int banked = totalPen - keptPen;        // banked this roll
                double w = prob6 * prob8 * prob10 * prob12;
                const std::vector<double>& DK = Dopt[stateIndex(keep6, keep8, keep10, keep12)];
                int mk = maxRemainOf[stateIndex(keep6, keep8, keep10, keep12)];
                for(int v = 0; v <= mk; ++v){
                    if(DK[v] > 0){
                        D[banked + v] += w * DK[v];
                    }
                }
            }
        }
    }
}

// =====================================================================
// (B) Win-value W(s): expected win-share finishing at score s, against
//     (N-1) i.i.d. opponents with score distribution D, fair tie-splitting.
// =====================================================================
static void computeW(const double* D, int N, double* W /*size SZ*/){
    // suffix sums: ge(s)=P(opp>s), eq(s)=P(opp=s)
    double ge = 0.0;
    for(int s = MAXS; s >= 0; --s){
        double eq  = D[s];
        double geq = ge + eq;                     // P(opp>=s)
        if(eq <= 0){
            W[s] = pow(ge, N - 1);                // no ties possible at s
        }else{
            W[s] = (pow(geq, N) - pow(ge, N)) / (N * eq);
        }
        ge = geq;
    }
}

// =====================================================================
// (C) Win-probability DP U(S,g).  Ugrid[sid] holds SZ doubles.
//     Only g in [0, maxBankedOf[sid]] are meaningful (others unreachable).
// =====================================================================
static double Ugrid[104][SZ];   // Ugrid[state][g] = win-probability DP table U(S,g)

// one worker: processes the d6 face-counts whose first count is face1 (grabbed via
// the shared atomic), accumulating  w * max_K U(K,g+banked)  into private acc[].
static void solveUWorker(int d6, int d8, int d10, int d12, int gmax,
                         std::atomic<int>* face1ctr, double* acc){
    int face1;
    while((face1 = face1ctr->fetch_add(1)) <= d6){
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
            double prob8  = d8  ? 1.0 / 8  : 1;
            double prob10 = d10 ? 1.0 / 10 : 1;
            double prob12 = d12 ? 1.0 / 12 : 1;
            for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
            for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
            for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
                int totalPen = (int)topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
                double w = prob6 * prob8 * prob10 * prob12;
                int candidateState[104], candidateBanked[104], candidateCount = 0;
                for(int try6  = 0; try6  <= d6;  ++try6)
                for(int try8  = 0; try8  <= d8;  ++try8)
                for(int try10 = 0; try10 <= d10; ++try10)
                for(int try12 = 0; try12 <= d12; ++try12){
                    if(try6 == d6 && try8 == d8 && try10 == d10 && try12 == d12){
                        continue;
                    }
                    int kept = (int)topPenSum6[try6] + (try8 ? pen8 : 0) + (try10 ? pen10 : 0) + (try12 ? pen12 : 0);
                    candidateState[candidateCount]  = stateIndex(try6, try8, try10, try12);
                    candidateBanked[candidateCount] = totalPen - kept;
                    ++candidateCount;
                }
                for(int g = 0; g <= gmax; ++g){
                    double best = -1e300;
                    for(int c = 0; c < candidateCount; ++c){
                        double v = Ugrid[candidateState[c]][g + candidateBanked[c]];
                        if(v > best){
                            best = v;
                        }
                    }
                    acc[g] += w * best;
                }
            }
        }
    }
}

static void solveU(const double* W){
    int e = stateIndex(0, 0, 0, 0);
    for(int g = 0; g <= maxBankedOf[e]; ++g){
        Ugrid[e][g] = W[g];                       // U(empty,g)=W(g)
    }
    for(int totalDice = 1; totalDice <= 15; ++totalDice)
    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice){
            continue;
        }
        int sid  = stateIndex(d6, d8, d10, d12);
        int gmax = maxBankedOf[sid];
        int nth  = (d6 >= 4) ? numThreads : 1;     // small states: not worth threading
        std::atomic<int> face1ctr(0);
        std::vector<std::vector<double>> accs(nth, std::vector<double>(gmax + 1, 0.0));
        if(nth == 1){
            solveUWorker(d6, d8, d10, d12, gmax, &face1ctr, accs[0].data());
        }else{
            std::vector<std::thread> th;
            for(int t = 0; t < nth; ++t){
                th.emplace_back(solveUWorker, d6, d8, d10, d12, gmax, &face1ctr, accs[t].data());
            }
            for(auto& x : th){
                x.join();
            }
        }
        for(int g = 0; g <= gmax; ++g){
            double s = 0;
            for(int t = 0; t < nth; ++t){
                s += accs[t][g];
            }
            Ugrid[sid][g] = s;
        }
    }
}

// best-response action at (S,g) for a given roll (uses current Ugrid).
static inline void brAction(int d6, int d8, int d10, int d12, const double* topPenSum6,
                            int pen8, int pen10, int pen12, int g,
                            int& keep6, int& keep8, int& keep10, int& keep12, int& banked){
    int totalPen = (int)topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
    double best = -1e300;
    keep6 = keep8 = keep10 = keep12 = 0;
    banked = 0;
    for(int try6  = 0; try6  <= d6;  ++try6)
    for(int try8  = 0; try8  <= d8;  ++try8)
    for(int try10 = 0; try10 <= d10; ++try10)
    for(int try12 = 0; try12 <= d12; ++try12){
        if(try6 == d6 && try8 == d8 && try10 == d10 && try12 == d12){
            continue;
        }
        int kept = (int)topPenSum6[try6] + (try8 ? pen8 : 0) + (try10 ? pen10 : 0) + (try12 ? pen12 : 0);
        int b = totalPen - kept;
        double v = Ugrid[stateIndex(try6, try8, try10, try12)][g + b];
        if(v > best){
            best = v;
            keep6  = try6;
            keep8  = try8;
            keep10 = try10;
            keep12 = try12;
            banked = b;
        }
    }
}

// =====================================================================
// (E-helper) Score distribution of the best-response policy currently in Ugrid.
//   Forward push of probability mass over (S,g) states, top-down by dice count.
//   Returns the final-score distribution (size SZ) into out[].
// =====================================================================
// worker: pushes mass from state S (rows of msrc) into a private delta grid.
static void pushMassWorker(int d6, int d8, int d10, int d12, int gmax, const double* msrc,
                           std::atomic<int>* face1ctr, double* delta /*104*SZ*/){
    int face1;
    while((face1 = face1ctr->fetch_add(1)) <= d6){
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
            double prob8  = d8  ? 1.0 / 8  : 1;
            double prob10 = d10 ? 1.0 / 10 : 1;
            double prob12 = d12 ? 1.0 / 12 : 1;
            double w = prob6 * prob8 * prob10 * prob12;
            for(int pen8  = 0; pen8  <= (d8  ? 7  : 0); ++pen8)
            for(int pen10 = 0; pen10 <= (d10 ? 9  : 0); ++pen10)
            for(int pen12 = 0; pen12 <= (d12 ? 11 : 0); ++pen12){
                for(int g = 0; g <= gmax; ++g){
                    double mass = msrc[g];
                    if(mass <= 0){
                        continue;
                    }
                    int keep6, keep8, keep10, keep12, b;
                    brAction(d6, d8, d10, d12, topPenSum6, pen8, pen10, pen12, g,
                             keep6, keep8, keep10, keep12, b);
                    delta[stateIndex(keep6, keep8, keep10, keep12) * SZ + (g + b)] += mass * w;
                }
            }
        }
    }
}

static void brScoreDist(double* out /*size SZ*/){
    static double m[104][SZ];
    for(int s = 0; s < 104; ++s){
        for(int g = 0; g < SZ; ++g){
            m[s][g] = 0.0;
        }
    }
    m[stateIndex(12, 1, 1, 1)][0] = 1.0;

    for(int totalDice = 15; totalDice >= 1; --totalDice)
    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
        if(d6 + d8 + d10 + d12 != totalDice){
            continue;
        }
        int sid  = stateIndex(d6, d8, d10, d12);
        int gmax = maxBankedOf[sid];
        bool any = false;
        for(int g = 0; g <= gmax; ++g){
            if(m[sid][g] > 0){
                any = true;
                break;
            }
        }
        if(!any){
            continue;
        }
        int nth = (d6 >= 4) ? numThreads : 1;
        std::atomic<int> face1ctr(0);
        std::vector<std::vector<double>> deltas(nth, std::vector<double>(104 * SZ, 0.0));
        if(nth == 1){
            pushMassWorker(d6, d8, d10, d12, gmax, m[sid], &face1ctr, deltas[0].data());
        }else{
            std::vector<std::thread> th;
            for(int t = 0; t < nth; ++t){
                th.emplace_back(pushMassWorker, d6, d8, d10, d12, gmax, m[sid], &face1ctr, deltas[t].data());
            }
            for(auto& x : th){
                x.join();
            }
        }
        for(int t = 0; t < nth; ++t){
            const double* d = deltas[t].data();
            for(int s = 0; s < 104; ++s){
                for(int g = 0; g < SZ; ++g){
                    m[s][g] += d[s * SZ + g];
                }
            }
        }
    }
    int e = stateIndex(0, 0, 0, 0);
    for(int g = 0; g < SZ; ++g){
        out[g] = m[e][g];
    }
}

// dot product of a finishing distribution with the win-value -> expected win-share.
static double winShare(const double* D, const double* W){
    double s = 0;
    for(int g = 0; g <= MAXS; ++g){
        s += D[g] * W[g];
    }
    return s;
}

// =====================================================================
// CSV artifacts: persist the strategy tables to disk.
//   A policy here is a VALUE table; the move is derived greedily from it.
// =====================================================================
static void dumpV(const char* path){
    FILE* f = fopen(path, "w");
    if(!f){
        perror(path);
        return;
    }
    fprintf(f, "# bitches: expected-optimal (mean-minimizing) value function V.\n");
    fprintf(f, "# state = dice still in hand (a6 d6 in 0..12; a8/a10/a12 each 0/1). V = optimal expected remaining score.\n");
    fprintf(f, "# OPTIMAL MOVE for a roll: keep the subset K of your dice maximizing (sum of kept penalties) - V[K].\n");
    fprintf(f, "a6,a8,a10,a12,V\n");
    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
        fprintf(f, "%d,%d,%d,%d,%.10g\n", d6, d8, d10, d12, V[stateIndex(d6, d8, d10, d12)]);
    }
    fclose(f);
    printf("  wrote %s (104 states)\n", path);
}

// dump the win-probability table currently in Ugrid (defines a g-dependent policy).
static void dumpUgrid(const char* path, const char* desc){
    FILE* f = fopen(path, "w");
    if(!f){
        perror(path);
        return;
    }
    fprintf(f, "# bitches competitive policy: %s\n", desc);
    fprintf(f, "# state = (a6,a8,a10,a12) dice in hand; g = penalty points already banked this game.\n");
    fprintf(f, "# U = expected win-share (probability of having the lowest score) from (state,g) under this policy.\n");
    fprintf(f, "# OPTIMAL MOVE at (state,g) for a roll: keep the subset K maximizing U[K][g + banked_penalty(K)].\n");
    fprintf(f, "# only reachable g (0..87-maxRemaining) are listed.\n");
    fprintf(f, "a6,a8,a10,a12,g,U\n");
    long rows = 0;
    for(int d6 = 0; d6 <= 12; ++d6)
    for(int d8 = 0; d8 <= 1; ++d8)
    for(int d10 = 0; d10 <= 1; ++d10)
    for(int d12 = 0; d12 <= 1; ++d12){
        int sid = stateIndex(d6, d8, d10, d12);
        for(int g = 0; g <= maxBankedOf[sid]; ++g){
            fprintf(f, "%d,%d,%d,%d,%d,%.10g\n", d6, d8, d10, d12, g, Ugrid[sid][g]);
            ++rows;
        }
    }
    fclose(f);
    printf("        wrote %s (%ld rows)\n", path, rows);
}

// =====================================================================
// Monte-Carlo tournament validation.
// =====================================================================
static inline uint64_t xorshift(uint64_t& rngState0, uint64_t& rngState1){
    uint64_t x = rngState0, y = rngState1;
    rngState0 = y;
    x ^= x << 23;
    rngState1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rngState1 + y;
}
static inline int rollDie(uint64_t& rngState0, uint64_t& rngState1, int sides){
    return (int)(((xorshift(rngState0, rngState1) >> 32) * (uint64_t)sides) >> 32);
}

// Play one full game with policy: comp==false -> expected-optimal (uses V);
// comp==true -> competitive best response in Ugrid (g-dependent). Returns score.
static int playGame(uint64_t& rngState0, uint64_t& rngState1, bool comp){
    int d6 = 12, d8 = 1, d10 = 1, d12 = 1, score = 0;
    while(d6 + d8 + d10 + d12 > 0){
        int h[6] = {0, 0, 0, 0, 0, 0};
        for(int j = 0; j < d6; ++j){
            h[rollDie(rngState0, rngState1, 6)]++;
        }
        double topPenSum6[13];
        {
            int faceCount[6] = {h[5], h[4], h[3], h[2], h[1], h[0]};
            buildTopPenSums(faceCount, topPenSum6);
        }
        int pen8  = d8  ? rollDie(rngState0, rngState1, 8)  : 0;
        int pen10 = d10 ? rollDie(rngState0, rngState1, 10) : 0;
        int pen12 = d12 ? rollDie(rngState0, rngState1, 12) : 0;
        int keep6, keep8, keep10, keep12;
        if(comp){
            int b;
            brAction(d6, d8, d10, d12, topPenSum6, pen8, pen10, pen12, score,
                     keep6, keep8, keep10, keep12, b);
        }else{
            optimalKeep(V, d6, d8, d10, d12, topPenSum6, pen8, pen10, pen12,
                  keep6, keep8, keep10, keep12);
        }
        int totalPen = (int)topPenSum6[d6] + (d8 ? pen8 : 0) + (d10 ? pen10 : 0) + (d12 ? pen12 : 0);
        int keptPen  = (int)topPenSum6[keep6] + (keep8 ? pen8 : 0) + (keep10 ? pen10 : 0) + (keep12 ? pen12 : 0);
        score += totalPen - keptPen;
        d6  = keep6;
        d8  = keep8;
        d10 = keep10;
        d12 = keep12;
    }
    return score;
}

// N seats: seat 0 = hero (heroComp), seats 1..N-1 = opponents (oppComp).
// NOTE: hero and opponents may share Ugrid only if they use the same field; if
// they differ, caller must serialize (this program runs one config at a time).
static double tournament(int N, bool heroComp, bool oppComp, long games, uint64_t seed){
    uint64_t rngState0 = seed * 0x9E3779B97F4A7C15ULL + 1;
    uint64_t rngState1 = seed * 0xD1B54A32D192ED03ULL + 0x9E3779B9ULL;
    for(int w = 0; w < 8; ++w){
        xorshift(rngState0, rngState1);
    }
    double heroShare = 0;
    std::vector<int> score(N);
    for(long t = 0; t < games; ++t){
        score[0] = playGame(rngState0, rngState1, heroComp);
        for(int i = 1; i < N; ++i){
            score[i] = playGame(rngState0, rngState1, oppComp);
        }
        int best = score[0];
        for(int i = 1; i < N; ++i){
            if(score[i] < best){
                best = score[i];
            }
        }
        int cnt = 0;
        for(int i = 0; i < N; ++i){
            if(score[i] == best){
                cnt++;
            }
        }
        if(score[0] == best){
            heroShare += 1.0 / cnt;
        }
    }
    return heroShare / games;
}

// =====================================================================
int main(int argc, char** argv){
    setvbuf(stdout, NULL, _IONBF, 0);
    numThreads = (argc > 2) ? atoi(argv[2]) : (int)std::thread::hardware_concurrency();
    if(numThreads < 1){
        numThreads = 1;
    }
    int eqMaxN = (argc > 3) ? atoi(argv[3]) : 4;   // largest N to solve the equilibrium for (each ~min on 16 cores)
    printf("(threads: %d ; equilibrium up to N=%d)\n", numThreads, eqMaxN);
    auto t0 = std::chrono::steady_clock::now();
    initTables();
    solveV(V);
    scoreDistOptimal();

    const double* Dfull = Dopt[stateIndex(12, 1, 1, 1)].data();   // expected-optimal score distribution

    printf("===== strategy artifacts (CSV) =====\n");
    dumpV("optimal_policy.csv");                          // single-agent expected-optimal value table
    const int DUMP_MAXN = eqMaxN;                         // record competitive tables for N=2..maxN (the CLI arg)


    // ---- (A) report the expected-optimal score distribution ----
    double mean = 0, m2 = 0;
    for(int s = 0; s <= MAXS; ++s){
        mean += s * Dfull[s];
    }
    for(int s = 0; s <= MAXS; ++s){
        m2 += (s - mean) * (s - mean) * Dfull[s];
    }
    double sd = sqrt(m2);
    printf("===== (A) Expected-optimal policy: full score distribution =====\n");
    printf("  V(full)        = %.5f   (DP optimal expected score)\n", V[stateIndex(12, 1, 1, 1)]);
    printf("  mean of dist   = %.5f   (must match V)\n", mean);
    printf("  std deviation  = %.5f   variance = %.4f\n", sd, m2);
    printf("  P(score=0)     = %.4e  (1 in %.0f)\n", Dfull[0], 1.0 / Dfull[0]);
    {   // percentiles
        int qs[] = {1, 5, 10, 25, 50, 75, 90, 95, 99};
        double cum = 0;
        int qi = 0;
        int pct[9];
        for(int s = 0; s <= MAXS && qi < 9; ++s){
            cum += Dfull[s];
            while(qi < 9 && cum * 100.0 >= qs[qi]){
                pct[qi] = s;
                ++qi;
            }
        }
        printf("  percentiles    :");
        for(int i = 0; i < 9; ++i){
            printf(" p%d=%d", qs[i], pct[i]);
        }
        printf("\n");
    }
    // E[min of N i.i.d. optimal scores] -> Fetterman Fig.13-style baseline
    printf("  E[min score] for N players (all expected-optimal):\n     ");
    for(int N = 2; N <= 8; ++N){
        // P(min=s) = P(score>=s)^N - P(score>s)^N; tail = P(score>=s), starts at 1.
        double Emin = 0, tail = 1.0;
        for(int s = 0; s <= MAXS; ++s){
            double pge = tail - Dfull[s];        // P(score>s)
            Emin += s * (pow(tail, N) - pow(pge, N));
            tail = pge;
        }
        printf(" N=%d:%.3f", N, Emin);
    }
    printf("\n\n");

    // ---- (D) best response to the field of expected-optimal players ----
    printf("===== (D) Best response vs (N-1) expected-optimal opponents =====\n");
    printf("   (win-share 1/N is the symmetric baseline; >1/N means exploitation)\n");
    static double W[SZ];   // win-value table W(s) for the current field
    for(int N = 2; N <= 8; ++N){
        computeW(Dfull, N, W);
        solveU(W);
        double br  = Ugrid[stateIndex(12, 1, 1, 1)][0];   // best-response win-share
        double opt = winShare(Dfull, W);                  // expected-optimal's own win-share vs same field
        printf("  N=%d  baseline 1/N=%.4f   expected-optimal=%.5f   best-response=%.5f  (+%.2f%% rel)\n",
               N, 1.0 / N, opt, br, 100.0 * (br - 1.0 / N) / (1.0 / N));
        if(N <= DUMP_MAXN){
            char p[64];
            snprintf(p, sizeof p, "competitive_bestresponse_N%d.csv", N);
            char d[96];
            snprintf(d, sizeof d, "best response vs %d expected-optimal opponents", N - 1);
            dumpUgrid(p, d);
        }
    }
    printf("\n");

    // ---- (E) symmetric equilibrium via fictitious play ----
    printf("===== (E) Symmetric equilibrium (fictitious play with damping) =====\n");
    int eqNsAll[] = {2, 3, 4, 6, 8};
    static double Deq[SZ], Dnew[SZ];   // current equilibrium field, and best-response score dist
    for(int ni = 0; ni < 5; ++ni){
        int N = eqNsAll[ni];
        if(N > eqMaxN){
            break;
        }
        for(int g = 0; g < SZ; ++g){
            Deq[g] = Dfull[g];      // init field = expected-optimal
        }
        printf("  N=%d  fictitious play:\n", N);
        // Damped fictitious play. Converges tightly for N<=6; large N (>=8) converges
        // slowly and may hit MAXIT (a non-convergence warning is printed if so). A
        // decreasing step / Anderson acceleration would tighten large-N — see TODO.
        double alpha = 0.5;
        double delta = 0;
        int it = 0;
        const int MAXIT = 120;
        for(it = 0; it < MAXIT; ++it){
            auto ti = std::chrono::steady_clock::now();
            computeW(Deq, N, W);
            solveU(W);
            brScoreDist(Dnew);                       // score dist of best response to Deq
            delta = 0;
            for(int g = 0; g < SZ; ++g){
                delta += fabs(Dnew[g] - Deq[g]);
            }
            for(int g = 0; g < SZ; ++g){
                Deq[g] = (1 - alpha) * Deq[g] + alpha * Dnew[g];
            }
            double share = Ugrid[stateIndex(12, 1, 1, 1)][0];
            double its = std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count();
            printf("      iter %2d: L1 delta=%.2e  BR win-share=%.5f  (%.2fs)\n", it, delta, share, its);
            if(delta < 1e-7){
                break;
            }
        }
        // consistency: at the fixed point each symmetric player's share must be 1/N.
        computeW(Deq, N, W);
        solveU(W);
        double selfshare = Ugrid[stateIndex(12, 1, 1, 1)][0];   // BR vs (N-1) equilibrium opponents
        double naiveVsEq = winShare(Dfull, W);                  // expected-optimal vs (N-1) equilibrium opponents
        double eqmean = 0;
        for(int s = 0; s <= MAXS; ++s){
            eqmean += s * Deq[s];
        }
        double eqsd = 0;
        for(int s = 0; s <= MAXS; ++s){
            eqsd += (s - eqmean) * (s - eqmean) * Deq[s];
        }
        eqsd = sqrt(eqsd);
        if(it >= MAXIT){
            printf("      WARNING: did not converge in %d iters (L1 delta %.2e); "
                   "N=%d table is approximate.\n", MAXIT, delta, N);
        }
        printf("  N=%d  conv in %d iters (L1 delta %.1e)\n", N, it, delta);
        printf("        equilibrium score dist: mean=%.4f sd=%.4f  (vs optimal mean=%.4f sd=%.4f)\n",
               eqmean, eqsd, mean, sd);
        printf("        equilibrium self win-share=%.5f (should be 1/N=%.5f)\n", selfshare, 1.0 / N);
        printf("        expected-optimal player vs (N-1) equilibrium opponents = %.5f  (%.2f%% rel vs 1/N)\n",
               naiveVsEq, 100.0 * (naiveVsEq - 1.0 / N) / (1.0 / N));
        if(N <= DUMP_MAXN){
            char p[64];
            snprintf(p, sizeof p, "competitive_equilibrium_N%d.csv", N);
            char d[96];
            snprintf(d, sizeof d, "symmetric Nash equilibrium policy for %d players", N);
            dumpUgrid(p, d);
        }
    }
    printf("\n");

    auto t1 = std::chrono::steady_clock::now();
    printf("(DP+equilibrium compute time %.2f s)\n\n", std::chrono::duration<double>(t1 - t0).count());

    // ---- (F) Monte-Carlo tournament validation ----
    long G = (argc > 1) ? atol(argv[1]) : 4000000;
    printf("===== (F) Monte-Carlo tournament validation (%ld games/config) =====\n", G);
    for(int N : {2, 3, 4}){
        // hero = best response to expected-optimal field; opponents = expected-optimal.
        computeW(Dfull, N, W);
        solveU(W);
        double dpBR = Ugrid[stateIndex(12, 1, 1, 1)][0];
        double mcBR = tournament(N, /*heroComp=*/true, /*oppComp=*/false, G, 0xC0FFEEull + N);
        // sanity: all expected-optimal -> hero share ~ 1/N
        double mcAll = tournament(N, false, false, G, 0xBEEFull + N);
        printf("  N=%d  best-response vs optimal:  DP=%.5f  MC=%.5f  (diff %+.5f)\n", N, dpBR, mcBR, mcBR - dpBR);
        printf("        all expected-optimal:       1/N=%.5f  MC=%.5f\n", 1.0 / N, mcAll);
    }
    printf("\n(total time %.2f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
