ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := parboil-stencil
SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)
SRCS := $(SRC_DIR)/main.cc

common.h: $(SRC_DIR)/common.h
	cp $< $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl common.h

OPTS ?= -n 16 -t 4

include ../common.mk