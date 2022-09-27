#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <arm_neon.h>
#include <pybind11/pybind11.h>
#define NUM_THREADS 128
void num_vector_neon(float32_t *A, float32_t *B, float32_t *C, int len);
void csr_spmm_kernel(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
);
//torch::Tensor spmm_cuda(
//    torch::Tensor rowptr,
//    torch::Tensor colind,
//    torch::Tensor values,
//    torch::Tensor dense
//);

//torch::Tensor spmm_cuda_no_edge_value(
//    torch::Tensor rowptr,
//    torch::Tensor colind,
//    torch::Tensor dense
//);


// #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a cuda tensor")
// #define CHECK_CONIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONIGUOUS(x)
torch::Tensor spmm_cpu(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor values,
    torch::Tensor dense
) {
    const auto m = rowptr.size(0)-1;
    const auto k = dense.size(1); //稠密矩阵的列数
    auto devid = dense.device().index(); // index() ?
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU, devid); // ?
    auto out = torch::empty({m,k}, options);

    csr_spmm_kernel(
        m, k, rowptr.data_ptr<int>(), colind.data_ptr<int>(), values.data_ptr<float>(), dense.data_ptr<float>(), out.data_ptr<float>());
    return out;

}

void csr_spmm_kernel(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
){
    int N = B_ncols;
    int M = A_nrows;
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int m=0; m<M; m++){
        int row_start = A_csrRowPtr[m];
        int row_end = A_csrRowPtr[m+1];

        for(int idx=row_start; idx<row_end; idx++){
            int k=A_csrColInd[idx];
            // for(int n=0; n<N; n++){
            //     C[m*N+n] += val[idx]*B[k*N+n];
            // }
            // cblas_saxpy(n, val[idx], B+(k*N), 1, C+(m*N), 1);
//            num_vector_neon(B+(k*N), val+idx, C+(m*N), N);
            num_vector_neon(B_dnVal+(k*N), A_csrVal+idx, C_dnVal+(m*N), N);
        }
    }
}

void num_vector_neon(float32_t *A, float32_t *B, float32_t *C, int len){
    // num-vector mul c = c+a*b
    int len1 = 4*(len/4);
    int len2 = len - len1;

    float32x4_t a;
    float32_t b=*B;//number
    float32x4_t c;

    for(int i=0; i<len1; i+=4){
        a = vld1q_f32(A+i);
        c = vld1q_f32(C+i);
        c = vfmaq_n_f32(c, a, b);
        vst1q_f32(C+i, c);
    }

    for(int i=len1; i<len; i++){
        C[i] = C[i]+A[i]*b;
    }
    // b = vld1_f32(B);
}

torch::Tensor csr_spmm(
    torch::Tensor A_rowptr,
    torch::Tensor A_colind,
    torch::Tensor A_csrVal,
    torch::Tensor B
) {
//    assert(A_rowptr.device().type()==torch::kCUDA);
//    assert(A_colind.device().type()==torch::kCUDA);
//    assert(A_csrVal.device().type()==torch::kCUDA);
//    assert(B.device().type()==torch::kCUDA);
    assert(A_rowptr.is_contiguous());
    assert(A_colind.is_contiguous());
    assert(A_csrVal.is_contiguous());
    assert(B.is_contiguous());
    assert(A_rowptr.dtype()==torch::kInt32);
    assert(A_colind.dtype()==torch::kInt32);
    assert(A_csrVal.dtype()==torch::kFloat32);
    assert(B.dtype()==torch::kFloat32);
    return spmm_cpu(A_rowptr, A_colind, A_csrVal, B);
}


//torch::Tensor csr_spmm_no_edge_value(
//    torch::Tensor A_rowptr,
//    torch::Tensor A_colind,
//    torch::Tensor B
//) {
//    assert(A_rowptr.device().type()==torch::kCUDA);
//    assert(A_colind.device().type()==torch::kCUDA);
//    assert(B.device().type()==torch::kCUDA);
//    assert(A_rowptr.is_contiguous());
//    assert(A_colind.is_contiguous());
//    assert(B.is_contiguous());
//    assert(A_rowptr.dtype()==torch::kInt32);
//    assert(A_colind.dtype()==torch::kInt32);
//    assert(B.dtype()==torch::kFloat32);
//    return spmm_cuda_no_edge_value(A_rowptr, A_colind, B);
//}

//torch::Tensor csr2csc_cuda(
//    torch::Tensor csrRowPtr,
//    torch::Tensor csrColInd,
//    torch::Tensor csrVal,
//    torch::Tensor cscColPtr,
//    torch::Tensor cscRowInd
//);
//
//torch::Tensor csr2csc(
//    torch::Tensor rowptr,
//    torch::Tensor colind,
//    torch::Tensor colptr,
//    torch::Tensor rowind,
//    torch::Tensor csr_data
//) {
//    assert(rowptr.device().type()==torch::kCUDA);
//    assert(colind.device().type()==torch::kCUDA);
//    assert(colptr.device().type()==torch::kCUDA);
//    assert(rowind.device().type()==torch::kCUDA);
//    assert(csr_data.device().type()==torch::kCUDA);
//    assert(rowptr.is_contiguous());
//    assert(colind.is_contiguous());
//    assert(colptr.is_contiguous());
//    assert(rowind.is_contiguous());
//    assert(csr_data.is_contiguous());
//    assert(rowptr.dtype()==torch::kInt32);
//    assert(colind.dtype()==torch::kInt32);
//    assert(colptr.dtype()==torch::kInt32);
//    assert(rowind.dtype()==torch::kInt32);
//    assert(csr_data.dtype()==torch::kFloat32);
//    return csr2csc_cuda(rowptr, colind, csr_data, colptr, rowind);
//}

PYBIND11_MODULE(spmm, m) {
    m.doc() = "spmm in CSR format. csr_spmm is the kernel with edge value. csr2csc provides the format transformation";
    m.def("csr_spmm", &csr_spmm, "CSR SPMM");
//    m.def("csr_spmm_no_edge_value", &csr_spmm_no_edge_value, "CSR SPMM NO EDGE VALUE");
//    m.def("csr2csc", &csr2csc, "csr2csc");
}

