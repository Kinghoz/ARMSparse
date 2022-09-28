# ARMSparse
Massively Parallel Optimization for Sparse Matrix Multiplication on ARM Many-core CPU
## Abstract
ARMSparse is an optimized computing library for sparse matrix computing on ARM manycore CPU,which can be applied to accelerate GCN and other applications base on matrix manipulation.

## Get started
```
git clone git@github.com:Kinghoz/ARMSparse.git
```
## Prerequisites
- AArch64 architecture
- NUMA architecture
- armadillo-10.5.1
- eigen-3.3.9
- OpenBLAS
- OpenMP/Pthread
- pytorch
- dgl
- pytorch_geometric

## ARMSparse
example for compiling sdmm spmm spmv 
```
g++ -g -O3 SDMM.cpp -o SDMM -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -fopenmp
g++ -g -O3 SpMM.cpp -o SpMM -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -fopenmp
g++ -g -O3 SpMV.cpp -o SpMV -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -fopenmp
```
run sdmm spmm spmv
```
./SDMM 4096 4096 4096 32 0.1
./SpMM 4096 4096 4096 32
./SpMV 4096 4096 32
```

## NUMA-aware

## application on GCN
