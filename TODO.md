# TODO / open frontiers

## 1. Competitive N-player strategy ("lowest score wins")  — OPEN

Everything solved so far minimizes the **expected score** (single-agent). But the
actual game is won by having the **lowest score at the table**. That's a different
objective, and it's the main thing Fetterman (2022, §5.1) flagged and set aside.

### Why it's a different problem
- The payoff is `P(your score < every opponent's score)`, not `E[score]`. You don't
  care how low you go, only about beating the field.
- This rewards **variance management**: when you're likely behind, you should gamble
  (accept higher variance — e.g. hold dice longer chasing a great finish); when ahead,
  play safe. The expected-score-optimal policy ignores this entirely.
- For symmetric players all reasoning this way, it becomes a **game-theoretic
  equilibrium** (each player's best response depends on the others' policies), not a
  single-agent optimization.

### Concrete plan
1. **Heads-up vs a fixed opponent (easiest first step).** We already have the full
   score *distribution* of any policy from the DP machinery (extend the value DP to
   carry the distribution of final scores per state, not just the mean). Given an
   opponent's score distribution, compute the policy that maximizes `P(win)` /
   minimizes `P(loss)` by a DP whose state adds "points banked so far" (or "deficit
   to the target you must beat"). State space grows (hand-state × score-so-far) but
   scores are small integers, so it stays tractable.
2. **Best response to "everyone plays the expected-optimal policy."** Use that policy's
   score distribution as the field; solve the win-maximizing best response. Measure how
   much it beats naive expected-optimal play head-to-head and N-handed.
3. **Symmetric equilibrium.** Iterate best-response (fictitious play) until the policy is
   a fixed point — the true competitive strategy for N symmetric players.
4. **Validate** with a multi-player Monte-Carlo tournament (reuse `opt_mc*` harness;
   each seat plays its policy, count wins). Confirm the DP win-probabilities.

### Useful starting points already in the repo
- `exact_dp.cpp` — the 104-state value DP to extend into a **score-distribution DP**.
- `perfect.cpp` / `maxperfect.cpp` — examples of DPs over the same states optimizing a
  *probability* (P(score=0)) rather than a mean; the win-probability DP is the same
  shape with a richer terminal payoff.
- Fetterman §5.1 Fig. 13 — his simulated "average minimum winning score" for 2–8
  players (all using his heuristic) — a sanity baseline to compare against.

### Related literature
- arXiv:0912.5518 — *Optimal minimax strategy in a dice game* (competitive framing).
- arXiv:1405.7488 — *A finite exact algorithm to solve a dice game*.
- Neller & Presser — *Optimal play of the dice game Pig* (win-probability DP template).

---

## 2. Smaller follow-ups (nice-to-have)
- **Single-thread speed:** the bucketed-SIMD (`opt_bucket.cpp`) lost to per-move
  data-movement; a counting-sort / index-based variant or a hybrid that keeps games
  in registers while batching only the argmax might still recover some. Low priority —
  multithreading already gives ~19M games/s.
- **Variance / full distribution reporting** for the expected-optimal policy (the
  score-distribution DP from item 1 gives this for free).
- **Publish the optimal value table** `V` as a clean `optimal_policy.csv` artifact.
