ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
TESTS := \
    vecadd sgemm conv3 psort saxpy sfilter sgemm2 sgemm3 psum oclprintf \
    blueporcelain-vectoradd blueporcelain-vectordiv                      \
    blueporcelain-triangle-area blueporcelain-vectormma                   \
    blueporcelain-matrix-transpose blueporcelain-bitonic-sort             \
    blueporcelain-dot-product blueporcelain-reduction-sum                 \
    blueporcelain-reduction-max blueporcelain-nbody                       \
    blueporcelain-prefix-sum                                              \
    blueporcelain-matrix-inversion blueporcelain-radix-sort               \
    blueporcelain-merge-sort blueporcelain-nqueen                         \
    blueporcelain-nn-op-adaptiveavgpool2d blueporcelain-nn-op-batchnorm2d \
    blueporcelain-nn-op-batchnorm3d blueporcelain-nn-op-conv2d            \
    blueporcelain-nn-op-conv3d blueporcelain-nn-op-dropout                \
    blueporcelain-nn-op-embedding blueporcelain-nn-op-gelu                \
    blueporcelain-nn-op-gru blueporcelain-nn-op-hardsigmoid               \
    blueporcelain-nn-op-hardswish blueporcelain-nn-op-layernorm           \
    blueporcelain-nn-op-leakyrelu blueporcelain-nn-op-linear              \
    blueporcelain-nn-op-lstm blueporcelain-nn-op-maxpool2d                \
    blueporcelain-nn-op-relu blueporcelain-nn-op-sigmoid                  \
    blueporcelain-nn-op-silu blueporcelain-nn-op-softmax                  \
    blueporcelain-nn-op-tanh                                              \
    polybench-2mm polybench-3mm polybench-atax                          \
    polybench-convolution-2d polybench-convolution-3d                   \
    polybench-bicg polybench-mvt polybench-gemm polybench-gesummv       \
    polybench-syr2k polybench-syrk polybench-correlation                \
    polybench-covariance polybench-fdtd-2d polybench-gramschmidt        \
    polybench-doitgen polybench-gemver polybench-lu                     \
    polybench-adi polybench-jacobi-1d-imper polybench-jacobi-2d-imper   \
    shoc-bfs shoc-triad shoc-md5hash shoc-md shoc-reduction shoc-scan    \
    shoc-sort shoc-spmv shoc-stencil2d shoc-gemm                          \
    parboil-sgemm parboil-spmv parboil-stencil parboil-mri-q             \
    parboil-lbm parboil-bfs parboil-sad parboil-cutcp                          \
    parboil-tpacf parboil-histo parboil-mri-gridding                       \
    rodinia-srad-v2 rodinia-particlefilter-single rodinia-leukocyte        \
    ventus-conv ventus-gemm ventus-softmax                               \
    ventus-add ventus-relu ventus-batchnorm2d ventus-pool2d              \
    ventus-conv-fused ventus-conv-bn ventus-gemm-fused                   \
    ventus-mnist                                                          \
    ventus-mnist-conv-tiny ventus-mnist-conv-small                          \
    dotproduct transpose spmv stencil lbm nearn guassian kmeans         \
    blackscholes bfs copybuf codegen histogram atomicreduce             \
    pathfinder nw lud hotspot hotspot3D srad backprop streamcluster     \
    lavaMD btree cfd dwt2d myocyte heartwall hybridsort                  \
    image_copy image_linear image_int image_get \
    image_ff image_ff_linear image_ff_bgra image_hwsw

# --- common exclude list ---------------------------------------------
# histogram/atomicreduce/hybridsort use atomic_add (RVA amo*.w); run them only
# with the A extension (CONFIGS="-DVX_CFG_EXT_A_ENABLE"), so exclude them from
# the default sweep which builds for the no-atomics config.
#
# image_* need the OpenCL image feature, which requires an image-enabled
# PoCL-Vortex build (CL_DEVICE_IMAGE_SUPPORT + the vortex scalar image builtins).
# They are excluded from the default sweep and run via their own opencl.yaml
# block that pins the image-enabled POCL_PATH.
EXCLUDE := \
    parboil-bfs \
    parboil-tpacf \
    parboil-histo \
    parboil-mri-gridding \
    transpose \
    lbm       \
    copybuf   \
    histogram \
    atomicreduce \
    hybridsort \
    image_copy \
    image_linear \
    image_int \
    image_get \
    image_ff \
    image_ff_linear \
    image_ff_bgra \
    image_hwsw

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
