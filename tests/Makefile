ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: unittest kernel regression opencl opencl-unique riscv

unittest:
	$(MAKE) -C unittest

kernel:
	$(MAKE) -C kernel

regression:
	$(MAKE) -C regression

opencl:
ifneq ($(XLEN),64)
	$(MAKE) -C opencl
endif

opencl-unique:
	$(MAKE) -C opencl-unique

riscv:
	$(MAKE) -C riscv

clean:
	$(MAKE) -C unittest clean
	$(MAKE) -C kernel clean
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean
	$(MAKE) -C opencl-unique clean
	$(MAKE) -C riscv clean

.PHONY: all unittest kernel regression opencl opencl-unique riscv