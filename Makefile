# Build all solvers/simulators for "biscuits (a dice game)".
#   make            # build everything
#   make run        # print the headline numbers
#   make test       # assert the headline numbers (exits non-zero on regression)
#   make clean

CXX      ?= c++
# ARM (Apple Silicon) wants -mcpu=native; x86 wants -march=native.
ARCH     := $(shell uname -m)
ifeq ($(ARCH),arm64)
  ARCHFLAGS := -mcpu=native
else
  ARCHFLAGS := -march=native
endif
CXXFLAGS ?= -O3 $(ARCHFLAGS) -funroll-loops -fopenmp-simd -std=c++17
LDFLAGS  ?=

BINS := biscuits_sim exact_dp thr_dp policy_extract sep_strict \
        opt_mc opt_mc_fast opt_mc_mt opt_card opt_simd8 opt_bucket \
        perfect maxperfect competitive

all: $(BINS)

# Most programs share the core DP/state code in biscuits.h, so rebuild on header change.
$(BINS): biscuits.h

# opt_mc_mt and competitive need pthreads
opt_mc_mt: opt_mc_mt.cpp
	$(CXX) $(CXXFLAGS) -pthread -o $@ $<

competitive: competitive.cpp
	$(CXX) $(CXXFLAGS) -pthread -o $@ $<

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: exact_dp thr_dp perfect maxperfect competitive opt_mc_mt
	@sh test.sh

run: exact_dp opt_mc_mt perfect maxperfect
	@echo "== exact optimum ==";        ./exact_dp | head -1
	@echo "== optimal MC (all cores) =="; ./opt_mc_mt 50000000 | grep -E 'mean|games/s'
	@echo "== perfect-game odds ==";     ./perfect 20000000 | head -1
	@echo "== max perfect-game odds =="; ./maxperfect 20000000 | head -1

clean:
	rm -f $(BINS)
	rm -rf *.dSYM

.PHONY: all run test clean
