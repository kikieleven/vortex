ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rodinia-particlefilter-single

SRC_DIR := $(VORTEX_HOME)/tests/opencl-unique/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# Nparticles must equal the work-group size (16): the kernels assume a single
# work-group, matching the upstream CUDA/OpenCL single-block launch.
OPTS ?= -x 32 -y 32 -f 4 -p 16

include ../common.mk