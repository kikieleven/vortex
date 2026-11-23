ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rodinia-srad-v2

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# rows/cols must be multiples of BLOCK_SIZE (4): the upstream CUDA v2 kernel
# tiles the image with integer blocks and has no partial-block guards.
OPTS ?= -r 32 -c 32 -n 2

include ../common.mk