CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -march=native -ffast-math \
             -Wall -Wextra -Wpedantic \
             -Iinclude -Ithird_party/eigen
LDFLAGS  :=

# ARM NEON intrinsics are auto-enabled by -march=native on the Vigx Ant-H1 Pro SoC.
# On macOS ARM (Apple M-series) same flag enables NEON; Eigen uses it automatically.

SRCS := src/model.cpp src/sensor.cpp src/reward.cpp src/trainer.cpp src/intent.cpp

all: olive_deploy olive_eval olive_tests

olive_deploy: $(SRCS) src/main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

olive_eval: $(SRCS) scripts/eval.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

olive_tests: $(SRCS) tests/test_olive.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

test: olive_tests
	./olive_tests

# Distill π0.5 → BaseController (synthetic teacher; no submodule required)
distill:
	python -m distillation.train --teacher pi0.5 --synthetic --steps 200 \
		--export checkpoints/base_controller_w0.bin

distill-test:
	python distillation/test_distill.py

eval: olive_eval
	./olive_eval 5000

clean:
	rm -f olive_deploy olive_eval olive_tests

.PHONY: all test eval clean distill distill-test
