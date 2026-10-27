# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# NVFP4EffMA: NVFP4 GEMM / GEMV / quantization for GPUs without native FP4 (sm_50 ... sm_90).
#   make tune       benchmark the kernel candidates on GPU $(GPU) and record the winners for its
#                   compute capability in src/nvfp4effma_tuned.h (other architectures keep defaults)
#   make lib        libnvfp4effma.a for every pre-FP4 architecture (SASS only, no PTX: never JIT-ed
#                   onto sm_100+, which has native FP4);  ARCHS=native builds only the local one
#   make test       bit-exact API tests + exhaustive core validation (GPU $(GPU))
#   make test-sm60  the GEMM / GEMV tests against a library built for sm_60 only: on an sm_61+ GPU this
#                   runs the paths of GPUs without DP4A (binary-compatible SASS) and the default tables
#   make test-sm75  the same against a library of compute_75 PTX only: on an sm_80+ GPU the driver JIT-compiles
#                   it, and the tensor-core kernels run their Turing variant (no cp.async, no mbarriers)
#   make bench      GEMM / GEMV / quantization throughput vs cuBLAS and memory bandwidth
#                   (bench/run_bench.sh: controlled run with board settings -> benches/<gpu>.md)
#   make install    header, static library and pkg-config file under PREFIX (default /usr/local)
# Python / PyTorch bindings: binds/python (pip install).
# Builds run in parallel by default: JOBS = min(cores, half of MemAvailable / JOB_MB), override with
# `make JOBS=n` (a plain -j on the command line is overridden).  Multi-architecture objects also
# compile their architectures in parallel (nvcc --threads) when there are more slots than objects.
CUDA_HOME ?= /usr/local/cuda
NVCC  ?= $(CUDA_HOME)/bin/nvcc
# host compiler for nvcc; g++-12 if present (CUDA 12.0-12.3 do not accept GCC 13), else nvcc's default
CCBIN ?= $(shell command -v g++-12 2>/dev/null)
GPU   ?= 0
RUN    = CUDA_VISIBLE_DEVICES=$(GPU)
DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i $(GPU) 2>/dev/null | head -1 | tr -d '.')
TEST_ARCH ?= $(if $(DETECTED),$(DETECTED),61)
VERSION := $(shell sed -n 's/^\#define NVFP4EFFMA_VERSION "\(.*\)"/\1/p' include/nvfp4effma.h)
ALL_ARCHS = 50 52 53 60 61 62 70 72 75 80 86 87 89 90
ARCHS ?= $(ALL_ARCHS)
ifeq ($(ARCHS),native)
  override ARCHS := $(TEST_ARCH)
endif
FLAGS  = $(if $(CCBIN),-ccbin $(CCBIN)) -O3 -Xptxas -O3 -std=c++14 -Xcompiler -ffp-contract=off,-fPIC
GENCODE = $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))

# parallelism within half of the available RAM (JOB_MB: peak of one nvcc process, ~110 MB measured)
JOB_MB ?= 512
NPROC  := $(shell nproc 2>/dev/null || echo 4)
MEM_MB := $(shell awk '/^MemAvailable:/ {print int($$2 / 1024)}' /proc/meminfo 2>/dev/null)
JOBS   ?= $(shell n=$$(( $(or $(MEM_MB),4096) / 2 / $(JOB_MB) )); [ $$n -gt $(NPROC) ] && n=$(NPROC); [ $$n -lt 1 ] && n=1; echo $$n)
MAKEFLAGS += -j$(JOBS)

CORE_H = include/nvfp4effma.h src/nvfp4effma.cuh src/nvfp4effma_tuned.h
GEMM_H = $(CORE_H) src/gemm_nvfp4effma.cuh
TC_H = $(GEMM_H) src/tensors/gemm_tc.cuh src/tensors/gemv_tc.cuh src/tensors/image.cuh src/tensors/mma.cuh
GEMV_H = $(CORE_H) src/gemv_nvfp4effma.cuh
QUANT_H = $(CORE_H) src/quant_nvfp4effma.cuh

# kernel configurations the library instantiates: defaults + those used by tuned entries
TUNED_IDS = $(shell sed -n 's/^\#define NVFP4A_$(1)_USED(X)//p' src/nvfp4effma_tuned.h | grep -o '[0-9]\+')
GEMM_IDS  = 0 1 $(call TUNED_IDS,GEMM)
GEMV_IDS  = 0 $(call TUNED_IDS,GEMV)
# one object per kernel instance -- block scale formats F (src/nvfp4effma.cuh: kF_*) are template parameters,
# all 16 for NVFP4 B, the 4 without B bits for the other B dtypes -- so the build runs in parallel: GEMV per
# config x x dtype x F, the FFMA GEMM per config x dense B dtype x F plus its NVFP4 / INT8 B kernels below
# sm_61 (0_0, 0_4), the DP4A GEMM per B dtype x F, the quantization per dense dtype x rowwise / columnwise x
# tile rows (dtype codes: NVFP4 0, BF16 1, FP16 2, FP32 3, INT8 4)
F_ALL = $(shell seq 0 15)
F_A   = 0 1 2 3
EX_ALL = 1 2 3 4 5
INST = $(foreach i,$(GEMV_IDS),$(foreach t,1 4,$(foreach x,$(EX_ALL),inst_gemvx_$(i)_$(t)_$(x)))) \
       $(foreach x,$(EX_ALL),$(foreach i,$(GEMM_IDS),inst_gemmx_$(i)_1_$(x)) inst_gemmx_0_4_$(x) inst_dp4ax_$(x)) \
       $(foreach x,$(EX_ALL),inst_gtcx_4_$(x) inst_gtcx_1_$(x) inst_tcx_4_$(x) inst_tcx_1_$(x)) \
       $(foreach i,$(GEMV_IDS),$(foreach f,$(F_ALL),inst_gemv_$(i)_0_$(f)) $(foreach t,1 2 3 4,$(foreach f,$(F_A),inst_gemv_$(i)_$(t)_$(f)))) \
       $(foreach i,$(GEMM_IDS),$(foreach t,1 2 3,$(foreach f,$(F_A),inst_gemm_$(i)_$(t)_$(f)))) \
       $(foreach f,$(F_ALL),inst_gemm_0_0_$(f) inst_dp4a_0_$(f)) $(foreach f,$(F_A),inst_gemm_0_4_$(f) inst_dp4a_4_$(f)) \
       $(foreach f,0 2 8 10,inst_tc_0_$(f)) $(foreach f,0 2,inst_tc_4_$(f) inst_tc_1_$(f)) \
       $(foreach f,0 2 4 6 8 10 12 14 16 18 24 26,inst_gtc_0_$(f)) $(foreach f,0 2,inst_gtc_4_$(f) inst_gtc_1_$(f)) \
       $(foreach t,1 2 3,$(foreach c,0 1,inst_quant_$(t)_$(c)_1 inst_quant_$(t)_$(c)_16))
OBJS = nvfp4effma gemm_nvfp4effma gemv_nvfp4effma tc_image $(INST)
NVCC_THREADS := $(shell t=$$(( $(JOBS) / $(words $(OBJS)) )); [ $$t -lt 1 ] && t=1; a=$(words $(ARCHS)); [ $$t -gt $$a ] && t=$$a; echo $$t)
word_ = $(word $(1),$(subst _, ,$(2)))

all: lib test-bin bench-bin

# ---- library: all pre-FP4 architectures (out/obj) and the local one (out/native) ---------------
define LIBRARY  # $(1) = object dir, $(2) = arch flags
$(1)/nvfp4effma.o: src/nvfp4effma.cu $(QUANT_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/gemm_nvfp4effma.o: src/gemm_nvfp4effma.cu $(TC_H) src/gemv_nvfp4effma.cuh include/nvfp4effma_items.h | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/gemv_nvfp4effma.o: src/gemv_nvfp4effma.cu $(GEMV_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/inst_gemmx_%.o: src/gemm_nvfp4effma_inst.cu $(GEMM_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_ID=$$(call word_,1,$$*) -DNVFP4A_TB=$$(call word_,2,$$*) -DNVFP4A_EX=$$(call word_,3,$$*) -c -o $$@ $$<
$(1)/inst_dp4ax_%.o: src/gemm_dp4a_inst.cu $(GEMM_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_EX=$$* -c -o $$@ $$<
$(1)/inst_gemm_%.o: src/gemm_nvfp4effma_inst.cu $(GEMM_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_ID=$$(call word_,1,$$*) -DNVFP4A_TB=$$(call word_,2,$$*) -DNVFP4A_F=$$(call word_,3,$$*) -c -o $$@ $$<
$(1)/inst_dp4a_%.o: src/gemm_dp4a_inst.cu $(GEMM_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_TB=$$(call word_,1,$$*) -DNVFP4A_F=$$(call word_,2,$$*) -c -o $$@ $$<
$(1)/tc_image.o: src/tensors/image.cu $(TC_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -c -o $$@ $$<
$(1)/inst_tcx_%.o: src/tensors/gemm_tc_inst.cu $(TC_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_TB=$$(call word_,1,$$*) -DNVFP4A_EX=$$(call word_,2,$$*) -c -o $$@ $$<
$(1)/inst_tc_%.o: src/tensors/gemm_tc_inst.cu $(TC_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_TB=$$(call word_,1,$$*) -DNVFP4A_F=$$(call word_,2,$$*) -c -o $$@ $$<
$(1)/inst_gtcx_%.o: src/tensors/gemv_tc_inst.cu $(TC_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_TB=$$(call word_,1,$$*) -DNVFP4A_EX=$$(call word_,2,$$*) -c -o $$@ $$<
$(1)/inst_gtc_%.o: src/tensors/gemv_tc_inst.cu $(TC_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_TB=$$(call word_,1,$$*) -DNVFP4A_F=$$(call word_,2,$$*) -c -o $$@ $$<
$(1)/inst_quant_%.o: src/quant_nvfp4effma_inst.cu $(QUANT_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_QT=$$(call word_,1,$$*) -DNVFP4A_QCOL=$$(call word_,2,$$*) -DNVFP4A_QTILE=$$(call word_,3,$$*) -c -o $$@ $$<
$(1)/inst_gemvx_%.o: src/gemv_nvfp4effma_inst.cu $(GEMV_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_ID=$$(call word_,1,$$*) -DNVFP4A_TX=$$(call word_,2,$$*) -DNVFP4A_F=0 -DNVFP4A_EX=$$(call word_,3,$$*) -c -o $$@ $$<
$(1)/inst_gemv_%.o: src/gemv_nvfp4effma_inst.cu $(GEMV_H) | $(1)
	$(NVCC) $(FLAGS) $(2) -DNVFP4A_ID=$$(call word_,1,$$*) -DNVFP4A_TX=$$(call word_,2,$$*) -DNVFP4A_F=$$(call word_,3,$$*) -c -o $$@ $$<
$(1)/libnvfp4effma.a: $(foreach o,$(OBJS),$(1)/$(o).o)
	rm -f $$@ && ar rcs $$@ $$^
$(1):
	mkdir -p $$@
endef
# multi-architecture objects (binds/python builds its own copy for the architectures it targets)
OBJDIR ?= out/obj
ifneq ($(OBJDIR),out/native)
$(eval $(call LIBRARY,$(OBJDIR),$(GENCODE) --threads $(NVCC_THREADS)))
endif
$(eval $(call LIBRARY,out/native,-arch=sm_$(TEST_ARCH)))
$(eval $(call LIBRARY,out/sm60,-arch=sm_60))
$(eval $(call LIBRARY,out/sm75,-arch=compute_75 -code=compute_75))
out/libnvfp4effma.a: $(OBJDIR)/libnvfp4effma.a
	cp $< $@
lib: out/libnvfp4effma.a

# ---- tuning, tests, benchmarks (local architecture) -------------------------------------------
NATIVE = out/native/libnvfp4effma.a
NATIVE_FLAGS = $(FLAGS) -arch=sm_$(TEST_ARCH)
out/tune: tune/tune.cu $(GEMM_H) $(GEMV_H) | out/native
	$(NVCC) $(NATIVE_FLAGS) -o $@ $<
tune: out/tune                   ## writes src/nvfp4effma_tuned.h for this GPU; rebuild the library afterwards
	$(RUN) ./out/tune src/nvfp4effma_tuned.h

TESTS   = test_gemm test_gemv test_gemv_rows test_epilogue test_nvfp4effma test_images
BENCHES = bench_gemm bench_gemv bench_quant bench_epilogue
$(addprefix out/,$(TESTS)): out/%: tests/%.cu tests/test_util.h $(NATIVE)
	$(NVCC) $(NATIVE_FLAGS) -o $@ $< $(NATIVE)
# the work items through an NVRTC-compiled executor (the library's headers from this tree)
ITEM_TEST_H = tests/test_util.h include/nvfp4effma_items.h include/nvfp4effma_items.cuh $(TC_H) $(GEMV_H)
ITEM_TEST_FLAGS = -DNVFP4A_ROOT='"$(CURDIR)"' -DNVFP4A_CUDA_INCLUDE='"$(CUDA_HOME)/include"'
out/test_items: tests/test_items.cu $(ITEM_TEST_H) $(NATIVE)
	$(NVCC) $(NATIVE_FLAGS) $(ITEM_TEST_FLAGS) -o $@ $< $(NATIVE) -lnvrtc -lcuda
out/sm60/test_items: tests/test_items.cu $(ITEM_TEST_H) out/sm60/libnvfp4effma.a
	$(NVCC) $(FLAGS) -arch=sm_60 $(ITEM_TEST_FLAGS) -o $@ $< out/sm60/libnvfp4effma.a -lnvrtc -lcuda
out/sm75/test_items: tests/test_items.cu $(ITEM_TEST_H) out/sm75/libnvfp4effma.a
	$(NVCC) $(FLAGS) -arch=compute_75 -code=compute_75 $(ITEM_TEST_FLAGS) -o $@ $< out/sm75/libnvfp4effma.a -lnvrtc -lcuda
out/test_nvfp4effma_neg: tests/test_nvfp4effma.cu tests/test_util.h $(NATIVE)
	$(NVCC) $(NATIVE_FLAGS) -DNVFP4A_NEGATIVE_CONTROLS -o $@ $< $(NATIVE)
out/sm60/test_%: tests/test_%.cu tests/test_util.h out/sm60/libnvfp4effma.a
	$(NVCC) $(FLAGS) -arch=sm_60 -o $@ $< out/sm60/libnvfp4effma.a
out/sm75/test_%: tests/test_%.cu tests/test_util.h out/sm75/libnvfp4effma.a
	$(NVCC) $(FLAGS) -arch=compute_75 -code=compute_75 -o $@ $< out/sm75/libnvfp4effma.a
$(addprefix out/,$(BENCHES)): out/%: bench/%.cu bench/bench_util.h include/nvfp4effma.h $(NATIVE)
	$(NVCC) $(NATIVE_FLAGS) -o $@ $< $(NATIVE) -lcublas -lnvidia-ml

test-bin: $(addprefix out/,$(TESTS)) out/test_nvfp4effma_neg out/test_items
bench-bin: $(addprefix out/,$(BENCHES))

test: test-bin                   ## API bit-exact, exhaustive encoders, negative controls
	$(RUN) ./out/test_gemm
	$(RUN) ./out/test_gemv
	$(RUN) ./out/test_gemv_rows
	$(RUN) ./out/test_epilogue
	$(RUN) ./out/test_items
	$(RUN) ./out/test_images
	$(RUN) ./out/test_nvfp4effma
	$(RUN) ./out/test_nvfp4effma_neg
test-sm60: out/sm60/test_gemm out/sm60/test_gemv out/sm60/test_epilogue out/sm60/test_items
	$(RUN) ./out/sm60/test_gemm
	$(RUN) ./out/sm60/test_gemv
	$(RUN) ./out/sm60/test_epilogue
	$(RUN) ./out/sm60/test_items
test-sm75: out/sm75/test_gemm out/sm75/test_gemv out/sm75/test_epilogue out/sm75/test_items
	$(RUN) ./out/sm75/test_gemm
	$(RUN) ./out/sm75/test_gemv
	$(RUN) ./out/sm75/test_epilogue
	$(RUN) ./out/sm75/test_items
bench: bench-bin
	$(RUN) ./out/bench_gemm
	$(RUN) ./out/bench_gemv
	$(RUN) ./out/bench_quant

# ---- install (C / C++ projects: pkg-config --cflags --libs nvfp4effma) -------------------------
PREFIX ?= /usr/local
install: $(OBJDIR)/libnvfp4effma.a
	install -d "$(DESTDIR)$(PREFIX)/include" "$(DESTDIR)$(PREFIX)/lib/pkgconfig" \
	           "$(DESTDIR)$(PREFIX)/include/nvfp4effma/include" "$(DESTDIR)$(PREFIX)/include/nvfp4effma/src/tensors"
	install -m 644 include/nvfp4effma.h include/nvfp4effma_items.h "$(DESTDIR)$(PREFIX)/include/"
	install -m 644 include/nvfp4effma.h include/nvfp4effma_items.cuh "$(DESTDIR)$(PREFIX)/include/nvfp4effma/include/"
	install -m 644 $(sort $(filter src/%,$(CORE_H) $(GEMV_H) $(GEMM_H))) "$(DESTDIR)$(PREFIX)/include/nvfp4effma/src/"
	install -m 644 $(filter src/tensors/%,$(TC_H)) "$(DESTDIR)$(PREFIX)/include/nvfp4effma/src/tensors/"
	{ sed -n '1,2p' include/nvfp4effma_items.cuh; printf '#include "nvfp4effma/include/nvfp4effma_items.cuh"\n'; } \
	    > "$(DESTDIR)$(PREFIX)/include/nvfp4effma_items.cuh"
	chmod 644 "$(DESTDIR)$(PREFIX)/include/nvfp4effma_items.cuh"
	install -m 644 $(OBJDIR)/libnvfp4effma.a "$(DESTDIR)$(PREFIX)/lib/"
	printf 'prefix=%s\nincludedir=$${prefix}/include\nlibdir=$${prefix}/lib\n\nName: nvfp4effma\nDescription: NVFP4 GEMM / GEMV / quantization for GPUs without native FP4\nVersion: $(VERSION)\nCflags: -I$${includedir} -I$(CUDA_HOME)/include\nLibs: -L$${libdir} -lnvfp4effma -L$(CUDA_HOME)/lib64 -lcudart -lstdc++\n' \
	       '$(PREFIX)' > "$(DESTDIR)$(PREFIX)/lib/pkgconfig/nvfp4effma.pc"

test-install: $(NATIVE)
	$(MAKE) install OBJDIR=out/native PREFIX="$(CURDIR)/out/install-test"
	$(NVCC) $(NATIVE_FLAGS) -DNVFP4A_ROOT='"$(CURDIR)/out/install-test"' -DNVFP4A_CUDA_INCLUDE='"$(CUDA_HOME)/include"' \
	    -o out/test_items_installed tests/test_items.cu out/install-test/lib/libnvfp4effma.a -lnvrtc -lcuda
	$(RUN) ./out/test_items_installed --coherence-only

clean:
	rm -rf out/obj out/native out/sm60 out/sm75 out/libnvfp4effma.a out/tune \
	       $(addprefix out/,$(TESTS) $(BENCHES)) out/test_nvfp4effma_neg out/test_items out/test_items_installed

.PHONY: all lib tune test-bin bench-bin test test-sm60 test-sm75 test-install bench install clean
