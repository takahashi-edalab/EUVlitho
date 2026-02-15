CUDADIR      ?= /gs/fs/tga-eda-takahashi/openblas/cuda
OPENBLASDIR  ?= /gs/fs/tga-eda-takahashi/openblas/openblas
EIGENDIR  ?= /gs/fs/tga-eda-takahashi/openblas/eigen-3.4.0

CPP           = g++
LD            = g++

CFLAGS        = -Wall -fopenmp -O3 -mavx512f -funroll-loops -mfma
#CFLAGS        = -Wall -fopenmp -O3 -march=znver3 -funroll-loops

LDFLAGS       = -Wall -fopenmp

SOURCE = m3d.cpp
OBJECT = $(SOURCE:.cpp=.o)

CUDA_LIBS = -L$(CUDADIR)/lib64 -lcudart -lcufft -lcusolver -lcublas
CUDA_INCLUDE = -I$(CUDADIR)/include
OPENBLAS_LIBS = -L$(OPENBLASDIR)/lib -lopenblas
OPENBLAS_INCLUDE = -I$(OPENBLASDIR)/include
EIGEN_INCLUDE = -I$(EIGENDIR)

.SUFFIXES:

m3d.out: $(OBJECT)
	$(LD) $(LDFLAGS) -o $@ $^ $(CUDA_LIBS) $(OPENBLAS_LIBS)

$(OBJECT):$(SOURCE)
	$(CPP) $(CFLAGS) $(CUDA_INCLUDE) $(EIGEN_INCLUDE) $(OPENBLAS_INCLUDE) -c -o $@ $<

