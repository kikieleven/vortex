ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := shoc-md5hash
SRC_DIR := $(VORTEX_HOME)/tests/opencl-unique/$(PROJECT)
SRCS := $(SRC_DIR)/main.cc

md5_core.h: $(SRC_DIR)/md5_core.h
	cp $< $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl md5_core.h

OPTS ?= -l 3 -v 16

include ../common.mk
