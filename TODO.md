# TODO / open frontiers

## 1. Competitive N-player strategy ("lowest score wins")  — DONE

Solved in `competitive.cpp` (exact DP, Monte-Carlo-validated). All four steps of
the original plan are implemented:

1. **Score-distribution DP.** The value DP is extended to carry the full
   distribution of final scores per state. For the expected-optimal policy: mean
   8.088 (= V), sd 3.57, median 8, p95 = 15, P(0) = 0.161%.
2. **Win-value `W(s)` + win-probability DP `U(S,g)`.** State adds "points banked so
   far" `g`; terminal payoff is the expected win-*share* against a field with a
   known score distribution,
   `W(s) = (P(opp>=s)^N - P(opp>s)^N) / (N·P(opp=s))` (fair tie-splitting).
   The maximizing action depends on `g` — the variance management the mean-optimal
   policy lacks. `U(full,0)` = win probability.
3. **Best response to expected-optimal field** beats the 1/N baseline, edge growing
   with table size: +0.4% (N=2), +3.8% (N=4), +7.9% (N=6), +11.8% (N=8) relative.
4. **Symmetric Nash equilibrium** via fictitious play (damped, converges to L1<1e-7
   for N<=6; N=8 converges slowly). Equilibrium players accept more variance
   (sd 3.57→4.19) and a slightly worse expected score (8.09→8.49) to win more often;
   each player's share is exactly 1/N at the fixed point (5-digit check). A naive
   expected-optimal player in an equilibrium field loses up to ~11% below 1/N (N=8).
5. **Validation** via multi-player Monte-Carlo tournament reproduces the DP
   win-shares (e.g. N=4 best-response DP 0.2595 vs MC 0.2615).

Notes / possible refinements (low priority):
- The two `(S,g)` DP passes are multithreaded but still ~10 s/iter on 16 cores;
  the full N=2..8 equilibrium sweep is ~25 min. Default run goes to N=4 (~10 min);
  pass a larger `maxN` arg for N=6/8. A faster equilibrium (better damping /
  Anderson acceleration, or pruning dominated kept-sets in the inner argmax) would
  help, and would let N=8 converge tightly.
- Fetterman §5.1 Fig. 13 baseline ("average minimum winning score", all-heuristic)
  is reproduced as `E[min score]` for N=2..8 (6.10 down to 3.60 under expected-
  optimal play) for comparison.

---

## 2. Smaller follow-ups (nice-to-have)
- **Single-thread speed:** the bucketed-SIMD (`opt_bucket.cpp`) lost to per-move
  data-movement; a counting-sort / index-based variant or a hybrid that keeps games
  in registers while batching only the argmax might still recover some. Low priority —
  multithreading already gives ~19M games/s.
- ~~**Publish the optimal value table** `V` as a clean `optimal_policy.csv` artifact.~~
  DONE — `competitive` writes `optimal_policy.csv` (the `V` table) plus, per player
  count N=2,3,4, `competitive_equilibrium_N{N}.csv` and
  `competitive_bestresponse_N{N}.csv` (the `U(S,g)` win-probability tables).

### Related literature
- arXiv:0912.5518 — *Optimal minimax strategy in a dice game* (competitive framing).
- arXiv:1405.7488 — *A finite exact algorithm to solve a dice game*.
- Neller & Presser — *Optimal play of the dice game Pig* (win-probability DP template).
