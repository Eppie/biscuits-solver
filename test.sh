#!/usr/bin/env sh
# Small smoke test for the "biscuits" solvers: assert the headline numbers so a
# refactor that silently breaks the math gets caught. Run `make test` (builds
# first) or `sh test.sh`. Exits non-zero if any check fails.
#
# These are the repo's correctness story: the exact DP values, and the fact that
# the Monte-Carlo simulator's mean agrees with the DP optimum.

set -u
cd "$(dirname "$0")"

fail=0
pass(){ printf '  ok    %s\n' "$1"; }
bad(){  printf '  FAIL  %s\n' "$1"; fail=1; }

# expect NAME WANT CMD...: run CMD and require its output to contain WANT.
expect(){
    name=$1; want=$2; shift 2
    out=$("$@" 2>/dev/null)
    case $out in
        *"$want"*) pass "$name" ;;
        *)         bad "$name  (expected to find: $want)" ;;
    esac
}

echo "Exact DP / probability invariants (deterministic):"
expect "optimal expected score = 8.087932"     "V(12,1,1,1) = 8.087932"                          ./exact_dp
expect "threshold-policy score = 8.529914"      "V(12,1,1,1) = 8.529914"                          ./thr_dp
expect "P(perfect) = 1.613223e-03"              "EXACT P(perfect | optimal play) = 1.613223e-03"  ./perfect 1
expect "max P(perfect) = 4.445638e-03"          "MAX P(perfect) = 4.445638e-03"                   ./maxperfect 1
expect "competitive score dist mean = 8.08793"  "mean of dist   = 8.08793" \
    sh -c './competitive 50 4 0 2>/dev/null | grep -m1 "mean of dist"'

echo "DP <-> Monte-Carlo agreement (optimal-policy mean within 0.02 of 8.0879):"
mc=$(./opt_mc_mt 3000000 4 2>/dev/null | sed -n 's/.*mean = \([0-9.][0-9.]*\).*/\1/p')
if [ -n "$mc" ] && awk "BEGIN{ d = $mc - 8.0879; if (d < 0) d = -d; exit !(d < 0.02) }"; then
    pass "opt_mc_mt mean = $mc  (~ 8.0879)"
else
    bad "opt_mc_mt mean = ${mc:-<none>}  (not within 0.02 of 8.0879)"
fi

echo
if [ "$fail" -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit "$fail"
