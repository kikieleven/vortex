ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := blueporcelain-matrix-transpose

SRC_DIR := $(VORTEX_HOME)/tests/opencl-unique/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -m 64 -n 128

include ../common.mk