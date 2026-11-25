ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := parboil-spmv
SRC_DIR := $(VORTEX_HOME)/tests/opencl-unique/$(PROJECT)
SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -n 256

include ../common.mk