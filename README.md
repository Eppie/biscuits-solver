# biscuits (a dice game) — solver & analysis

A complete analysis of the dice game **"biscuits"** (Glue Bunny Games): exact optimal
strategy, expected scores, perfect-game odds, and high-performance Monte-Carlo
simulators.

> **Note:** *biscuits* is Glue Bunny Games' family-friendly edition of their original
> game *bitches*; the two are mechanically identical, so this analysis applies to both.
> The retail *biscuits* set ships **11× d6 plus one custom six-sided "biscuits" die**,
> i.e. 12 six-sided dice in total — so the 12× d6 model below applies to both editions.

## The game
- **15 dice:** 12× d6, 1× d8, 1× d10, 1× d12.
- Each roll, reroll all dice in hand and **remove at least one** (up to all).
- A removed die scores **`penalty = max_face − shown_face`** (0 if removed at its max).
- Final score = sum of penalties. **Lowest score wins.** (You can never bank *zero*
  dice — that's the only constraint; banking everything at once is always legal.)

## Headline results

| Strategy | Expected score | How |
|---|---:|---|
| Idealized floor (unreachable) | 3.31 | parallel single-die optimal stopping |
| **Provably optimal** | **8.088** | exact 104-state DP, MC-validated |
| Best "decision-card" cutoff approximation | 8.116 | rank cutoffs extracted from optimal (`opt_card`) |
| Threshold heuristic (our forced rule) | 8.486 | exact DP = MC |
| Fetterman's published heuristic | 8.530 | exact DP = his reported 8.53 |
| "take exactly 1 die" greedy | 9.693 | exact DP = MC |

- **Optimal expected score = 8.0879** (exact DP; Monte-Carlo of the optimal policy
  averages 8.0872 over 30M games — agreement confirms both).
- The optimal action each roll: **keep the subset `K` of your dice maximizing
  `(sum of kept penalties) − V(K)`**, where `V` is the 104-state value function.
- **The optimum is NOT a per-die threshold:** 99 of 104 states are genuinely
  non-separable (only the 4 single-die endgames are separable). Whether to bank a
  given face depends on the *rest* of your hand. Fetterman's "Blackjack table" is a
  good approximation (~0.4 pts / 5% above optimal), not the true optimum.
- **Perfect game (score 0):** **0.161%** (≈1 in 620) under optimal (expected-score)
  play; **0.445%** (≈1 in 225) under the policy that *maximizes* perfect-game odds
  (≈ "bank every max-faced die," occasionally keeping one). This probability is
  strategy-*dependent*: Fetterman's 2022 paper guessed it "likely strategy-independent,"
  but his 2025 follow-up and this repo both show it is not (see *Relation to prior work*).

## Competitive play (lowest score wins, N players)

Everything above minimizes the **expected** score (single agent). The game is
actually won by the **lowest score at the table** — a different objective that
rewards *variance management* (gamble when behind, play safe when ahead) and, for
symmetric players, defines a **game-theoretic equilibrium**. `competitive.cpp`
solves all of this exactly (DP, not simulation) and Monte-Carlo-validates it.

- **Full score distribution** of the expected-optimal policy (not just the mean):
  mean **8.088** (= V, as it must), **sd 3.57**, median 8, p95 = 15, P(0) = 0.161%.
- **Win-value DP** `U(S,g)`: state = (hand-state, points banked so far `g`); payoff
  is the expected win-*share* against a field with a known score distribution. The
  optimal action now **depends on `g`** — exactly the variance management the
  mean-optimal policy lacks. `U(full,0)` is the player's win probability.
- **Best response to a table of expected-optimal players beats the 1/N baseline,
  and the edge grows with table size:**

  | players N | 1/N | best-response win-share | relative edge |
  |---:|---:|---:|---:|
  | 2 | 0.5000 | 0.5019 | +0.4% |
  | 4 | 0.2500 | 0.2595 | +3.8% |
  | 6 | 0.1667 | 0.1799 | +7.9% |
  | 8 | 0.1250 | 0.1397 | +11.8% |

- **Symmetric Nash equilibrium** (via fictitious play): the equilibrium policy
  **deliberately accepts more variance and a slightly worse expected score** to win
  more often. Equilibrium score sd rises 3.57 → 4.19 and mean drifts 8.09 → 8.49 as
  N goes 2 → 8. At equilibrium each player's share is exactly 1/N (a fixed-point
  check the solver confirms to 5 digits); a naive expected-optimal player dropped
  into an equilibrium field loses up to **~11% below its fair 1/N share** (N = 8).
- **Validation:** a multi-player Monte-Carlo tournament reproduces the DP win-shares
  (e.g. N = 4 best-response: DP 0.2595 vs MC 0.2615; all-optimal seats → 1/N).

**Recorded strategy tables (CSV).** Each run writes the full policies to disk:
`optimal_policy.csv` (the single-agent value table `V`), and, for each player count
N = 2,3,4, `competitive_equilibrium_N{N}.csv` (symmetric-equilibrium policy) and
`competitive_bestresponse_N{N}.csv` (best response to a naive field). A competitive
file lists `U(state,g)` for every reachable `(hand-state, points-banked-so-far g)`;
the move is `keep the subset K maximizing U[K][g+banked]` (just as `V` drives the
mean-optimal move). Bump the `maxN` arg to also emit N = 6/8 tables (long run).

## Relation to prior work
The prior analysis is Dave Fetterman's two notes on *bitches*, the original edition
of the same game:

- [*"Strategy for bitches (a dice game)"*](http://fettermania.com/math/bdice.pdf)
  (2022) — the expected-score problem. We reproduced his per-die thresholds (Fig. 7)
  exactly, confirmed his simulated 8.53 via exact DP, and **solved the true
  mean-optimum he left open** (he sketched the 104-state DP but didn't run it): 8.088.
- [*"BADG 2: A Perfect Game"*](https://www.fettermania.com/math/bdice2.pdf) (2025) —
  the perfect-game problem. His 2022 note had guessed the perfect-game probability was
  "likely strategy-independent"; this 2025 follow-up corrects that itself, proving the
  naive "always bank every max" policy is *not* optimal (§1.2, "Mistake 1"). Our
  `maxperfect` reaches the same conclusion independently, and its value
  (4.445638e-03) matches his 2025 optimum (0.0044456381) to 6 significant figures.
  Note `perfect` answers a different question — the perfect-game odds *under the
  mean-optimal policy* (1.613223e-03) — which is lower precisely because that policy
  banks sub-max dice.

The single-die sub-problem is the classic Cayley/optimal-stopping "die problem"; the
closest published academic analog is Wouter van Doorn's [*"On maximizing the number of
heads when you need to set aside at least one coin every round"*](https://arxiv.org/abs/2406.14700)
(arXiv:2406.14700, 2024) — a binary-outcome cousin of the same "bank ≥1 each round" structure.

## Files

- `biscuits.h` — **shared core** included by most programs: the state encoding
  (`stateIndex`), the exact value DP (`solveV`), the optimal per-roll action it
  induces (`optimalKeep`), and the d6 top-penalty prefix sums (`buildTopPenSums`).
  Single source of truth for the value function so it isn't re-implemented per file.

### Exact solvers (dynamic programming over the 104 hand-states)
- `exact_dp.cpp` — optimal value function `V`; prints optimal expected score (8.0879).
- `thr_dp.cpp` — exact expected value of the threshold (Blackjack-table) policy (8.530).
- `policy_extract.cpp` — extracts/analyzes per-state cutoffs; separability check.
- `sep_strict.cpp` — strict (tie-immune) separability test → 99/104 non-separable.
- `perfect.cpp` — exact P(perfect) under optimal play + MC validation.
- `maxperfect.cpp` — policy maximizing P(perfect) and its probability.
- `competitive.cpp` — **N-player "lowest score wins"**: score-distribution DP,
  win-probability DP `U(S,g)`, best response to expected-optimal field, symmetric
  Nash equilibrium (fictitious play), and MC tournament validation. Multithreaded.
  `./competitive [games] [threads] [maxN_equilibrium]` (default `4000000 <cores> 4`;
  pass a larger `maxN` for the full N=6/8 equilibrium — minutes per N).

### Monte-Carlo simulators
- `biscuits_sim.cpp` — **vectorized** (NEON/AVX) sim of the *take-1* and *threshold*
  strategies. ~24–26M games/s single-thread. `./biscuits_sim [games] [seed] [take1|threshold]`.
- `opt_mc.cpp` — optimal-policy MC, reference (mt19937 + sort). ~0.75M games/s.
- `opt_mc_fast.cpp` — optimal-policy MC, **branchless argmax + fast RNG**. ~1.7M games/s.
- `opt_mc_mt.cpp` — multithreaded optimal-policy MC. **~19M games/s / 216M moves/s** on 16 cores.
  `./opt_mc_mt [games] [threads]`.

### Optimization experiments (kept for the record; none beat `opt_mc_fast`)
- `opt_card.cpp` — rank-threshold "decision card" (proved non-exact: optimum isn't
  a per-rank threshold).
- `opt_simd8.cpp` — 8-wide over big-die configs (loses to wasted lanes).
- `opt_bucket.cpp` — SIMD-across-games via state bucketing (loses to per-move
  data-movement overhead). Correct, just slower.

## Build & run
```sh
make            # builds everything (auto-detects ARM -mcpu / x86 -march)
make run        # smoke test of the headline numbers
./opt_mc_mt 100000000        # optimal-policy MC, all cores
./biscuits_sim 50000000 1 threshold
```

## Performance notes (optimal-policy MC)
The optimal policy does a ~100-evaluation argmax against `V` every move, so it's
fundamentally heavier per move than the threshold sim's single compare. Single-thread
went 0.75M → 1.7M (2.3×) via fast RNG + a branchless argmax; further single-thread
vectorization doesn't pay (games diverge into different states every move, so SIMD
either wastes lanes or pays data-movement). Multithreading scales it linearly to ~19M
games/s. All speed variants reproduce the exact optimum (8.0879) to MC precision.
