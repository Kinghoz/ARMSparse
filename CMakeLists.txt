cmake_minimum_required(VERSION 3.17)
project(ArmSparse)

set(CMAKE_CXX_STANDARD 14)
FIND_PACKAGE(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(.)
add_executable(ArmSparse
        sparseMatrixAdd.cpp)
#add_executable(ArmSparse
#        datasets/suitesparse.cpp
#        datasets/test.cpp
#        arm_sparse_saxpy.c
#        armadillo_sdmm.cpp
#        armadillo_sdmm_suitesparse.cpp
#        armadillo_spmm.cpp
#        armadillo_spmm_suitesparse.cpp
#        armadillo_spmv.cpp
#        armsparse_spmv_suitsparse.cpp
#        compare.cpp
#        Eigen_sdmm_suitesparse.cpp
#        Eigen_sparseAdd.cpp
#        Eigen_spmm_suitesparse.cpp
#        Eigen_spmv_suitesparse.cpp
#        EigenSpMM.cpp
#        eigenSpmv.cpp
#        EigenTranspose.cpp
#        mkl_spmm.cpp
#        numaAwareSPDM.cpp
#        rowBaseSpmm.cpp
#        rowBaseSpmm.h
#        Sparse_Matrix_Assignment.c
#        sparseMatrixAdd.cpp
#        sparseTranspose.cpp
#        SpMM.cpp
#        SpMV.cpp
#        suitesparse.cpp
#        suitesparse_spmm.cpp)
