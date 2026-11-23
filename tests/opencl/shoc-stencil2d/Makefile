ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := shoc-stencil2d
SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)
SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -n 64 -t 2

include ../common.mk
