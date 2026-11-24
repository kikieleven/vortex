ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
TESTS := \
    saxpy oclprintf \
    blueporcelain-vectoradd blueporcelain-vectordiv \
    blueporcelain-triangle-area blueporcelain-vectormma \
    blueporcelain-matrix-transpose blueporcelain-bitonic-sort \
    blueporcelain-dot-product blueporcelain-reduction-max \
    blueporcelain-nbody blueporcelain-matrix-inversion \
    blueporcelain-nqueen blueporcelain-nn-op-adaptiveavgpool2d \
    blueporcelain-nn-op-batchnorm3d blueporcelain-nn-op-conv3d \
    blueporcelain-nn-op-dropout blueporcelain-nn-op-embedding \
    blueporcelain-nn-op-gelu blueporcelain-nn-op-gru \
    blueporcelain-nn-op-hardsigmoid blueporcelain-nn-op-hardswish \
    blueporcelain-nn-op-layernorm blueporcelain-nn-op-leakyrelu \
    blueporcelain-nn-op-linear blueporcelain-nn-op-lstm \
    blueporcelain-nn-op-sigmoid blueporcelain-nn-op-silu \
    blueporcelain-nn-op-tanh polybench-3mm \
    polybench-syr2k polybench-covariance \
    polybench-doitgen polybench-gemver \
    polybench-lu polybench-adi \
    shoc-bfs shoc-triad \
    shoc-md5hash shoc-md \
    shoc-reduction shoc-scan \
    shoc-sort shoc-spmv \
    shoc-stencil2d parboil-spmv \
    parboil-mri-q parboil-lbm \
    parboil-sad parboil-tpacf \
    parboil-histo parboil-mri-gridding \
    rodinia-srad-v2 rodinia-particlefilter-single \
    rodinia-leukocyte ventus-conv \
    ventus-gemm ventus-softmax \
    ventus-relu ventus-batchnorm2d \
    ventus-pool2d ventus-conv-fused \
    ventus-conv-bn ventus-gemm-fused \
    ventus-mnist ventus-mnist-conv-small \
    spmv nearn \
    kmeans blackscholes \
    copybuf atomicreduce \
    nw hotspot3D \
    srad backprop \
    btree cfd \
    dwt2d myocyte \
    heartwall image_copy \
    image_int image_hwsw
# --- common exclude list ---------------------------------------------
# Excluded from the default sweep for reasons other than duplication:
#   * atomic cases need the A extension (CONFIGS="-DVX_CFG_EXT_A_ENABLE"):
#     parboil-tpacf, parboil-histo, parboil-mri-gridding, atomicreduce
#   * image_* need the image-enabled PoCL build
#   * copybuf exercises buffer copies only (no kernel)
# The original tests/opencl also has lbm and codegen; both fail on Vortex 3.0
# ("clEnqueueNDRangeKernel" and "misaligned memory access"), so they are not
# part of this deduplicated set at all.
EXCLUDE := \
    parboil-tpacf parboil-histo \
    parboil-mri-gridding copybuf \
    atomicreduce image_copy \
    image_int image_hwsw
# --- per-backend exclude lists ---------------------------------------
EXCLUDE_simx   :=
EXCLUDE_rtlsim :=
EXCLUDE_opae   :=
EXCLUDE_xrt    :=

BACKENDS := simx rtlsim opae xrt

# --- aggregate targets ------------------------------------------------
.PHONY: all clean $(addprefix run-,$(BACKENDS))

ACTIVE_TESTS = $(filter-out $(EXCLUDE),$(TESTS))
backend_tests = $(filter-out $(EXCLUDE) $(EXCLUDE_$(1)),$(TESTS))

all:   $(addprefix build-,$(ACTIVE_TESTS))
clean: $(addprefix clean-,$(TESTS))

define backend_rule
.PHONY: run-$(1)
run-$(1): $$(addprefix run-$(1)-,$$(call backend_tests,$(1)))
endef
$(foreach b,$(BACKENDS),$(eval $(call backend_rule,$(b))))

# --- pattern recipes --------------------------------------------------
.PHONY: build-% clean-%

build-%:
	$(MAKE) -C $* all

clean-%:
	$(MAKE) -C $* clean

# `$*` is <driver>-<case>. Split off the driver (no hyphens) and keep the
# remainder as the case name: case names may contain hyphens (polybench-2mm,
# blueporcelain-nn-op-tanh), so taking the second word is wrong.
run-%:
	$(MAKE) -C $(patsubst $(firstword $(subst -, ,$*))-%,%,$*) run-$(firstword $(subst -, ,$*))
