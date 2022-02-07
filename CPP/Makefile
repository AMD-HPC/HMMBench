SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)

ROCM_PATH ?= /opt/rocm
HIPCC      = ${ROCM_PATH}/bin/hipcc

CXXFLAGS       = -O3 -std=c++11
CXXFLAGS      += --amdgpu-target=gfx90a

rocm: $(OBJ)

.cpp.o:
	$(HIPCC) $(CXXFLAGS) $(HIP_INC) $< -o $@
