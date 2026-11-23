ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rodinia-leukocyte

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# -s: square frame size (must be >= 45 so the GICOV interior is non-empty)
# -t: MGVF iterations (upstream default is 500; kept small for RTL simulation)
OPTS ?= -s 48 -t 20

include ../common.mk