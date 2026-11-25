ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := ventus-softmax

SRC_DIR := $(VORTEX_HOME)/tests/opencl-unique/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -m 8 -n 64 -c 8 -y 8 -x 8

include ../common.mk