ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := blueporcelain-nn-op-batchnorm3d

SRC_DIR := $(VORTEX_HOME)/tests/opencl-unique/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

common.h: $(SRC_DIR)/common.h
	cp $< $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl common.h

OPTS ?= -l2

include ../common.mk
