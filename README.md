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
Examples of SuiteSparse dataset evaluation
```
g++ -g -O3 suitesparse_sdmm.cpp -o suitesparse_sdmm -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -fopenmp
g++ -g -O3 suitesparse_spmm.cpp -o suitesparse_spmm -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -fopenmp

./suitesparse_sdmm 32
./suitesparse_spmm 32
```
Benchmark in ./benchmark/eigen and ./benchmark/armadillo

## NUMA-aware
```
cd numaAware
g++ -g -O3 numaAwareSDMM.cpp -o numaAwareSDMM -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -lpthread -lnuma -fopenmp
g++ -g -O3 numaAwareSpMM.cpp -o numaAwareSpMM -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -lpthread -lnuma -fopenmp
g++ -g -O3 numaAwareSpMV.cpp -o numaAwareSpMV -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -lpthread -lnuma -fopenmp

./numaAwareSDMM 4096 4096 4096 
./numaAwareSpMM 4096 4096 4096 
./numaAwareSDMV 4096 4096
```

## application on GCN
we integrate ARMSparse into pytorch and implement a graph convolution network with our contribution.
### Prerequisites
- pytorch
- dgl
- pytorch_geometric

Simple example for gcn_custom. You are able to switch the dataset to PubMed or cora and change the structure of network.
```
cd gcn/pytorch-custom
python gcn_custom.py --n-hidden=64
```
Example of GCN in PYG.
```
cd gcn/pytorch-custom
python gcn_pyg.py --n-hidden=64
```
Example of GCN in DGL
```
cd $(this-repo)/dgl-custom/benchmark
cd gcn
python gcn_dgl.py --gpu=0 --dataset=pubmed --n-hidden=64 --n-layers=1
```
