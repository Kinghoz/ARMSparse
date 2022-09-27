//g++ -g -O3 SpMM.cpp -o SpMM -I /home/nscc-gz/zhengj/opt/OpenBLAS/include -L /home/nscc-gz/zhengj/opt/OpenBLAS/lib -lopenblas -fopenmp
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>
#include <arm_neon.h>
#include <cmath>
using namespace std;
void clear(float* matrix, int m, int n);
void num_vector_neon(float32_t *A, float32_t *B, float32_t *C, int len);
void origin(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads);
int prepareSparse(float* matrix, int rows, int cols, float ratio);
void toCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val);
double getHighResolutionTime(void);
void mul(int m, int k, int n, float* A, float* B, float* C);
bool compare(int m,int n, float* C1, float* C2);
void toCSR2(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val);
void prepareSparse2(float* matrix, int rows, int cols,int* rowPtr, int* colInd, float* val, float ratio);
void rowBaseSpmm(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){
    // omp_set_num_threads(threads);
    // int threads = 8;
    // #pragma omp parallel for num_threads(threads) collapse(2)
    // omp_set_nested(8);
    // #pragma omp parallel for num_threads(threads)
    // #pragma omp parallel
    // #pragma omp single
    int n = N;
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        
        float32x4_t c0;
        int idx;
        float32x4_t _a, b0,b1,b2,b3;
        // int32x4_t _colind;
        for(idx=row_start; idx+4<=row_end; idx+=4){
            _a = vld1q_f32(val+idx);
            
            for(int j=0; j+4<=N; j+=4){
                
                
                // _colind = vld1q_s32(colInd+idx);
                c0 = vld1q_f32(C+m*N+j);

                b0 = vld1q_f32(B+colInd[idx]*N+j);
                b1 = vld1q_f32(B+colInd[idx+1]*N+j);
                b2 = vld1q_f32(B+colInd[idx+2]*N+j);
                b3 = vld1q_f32(B+colInd[idx+3]*N+j);

                c0 = vfmaq_laneq_f32(c0, b0, _a, 0);
                c0 = vfmaq_laneq_f32(c0, b1, _a, 1);
                c0 = vfmaq_laneq_f32(c0, b2, _a, 2);
                c0 = vfmaq_laneq_f32(c0, b3, _a, 3);
                vst1q_f32(C+m*N+j, c0);

            }
            // _a = vld1q_f32(val+idx);
            // _colind = vld1q_s32(colInd+idx);
            // for(int base=idx; idx<row_end; idx++){
            //     b0 = vld1q_f32(B+_colind[idx-base]*N+j);
            //     c0 = vfmaq_laneq_f32(c0, b0, _a, idx-base);
            // }
            
        }
    }

}
void rowBaseSpmm2(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){
    // omp_set_num_threads(threads);
    // int threads = 8;
    // #pragma omp parallel for num_threads(threads) collapse(2)
    // omp_set_nested(8);
    // #pragma omp parallel for num_threads(threads)
    // #pragma omp parallel
    // #pragma omp single
    int n = N;
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        
        for(int idx=row_start; idx<row_end; idx++){
            int k=colInd[idx];
            // for(int n=0; n<N; n++){
            //     C[m*N+n] += val[idx]*B[k*N+n]; 
            // }
            cblas_saxpy(n, val[idx], B+(k*N), 1, C+(m*N), 1);
            // num_vector_neon(B+(k*N), val+idx, C+(m*N), n);
        }      
    }

}
void rowBaseSpmm3(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){

    int n = N;
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        
        for(int idx=row_start; idx<row_end; idx++){
            int k=colInd[idx];
            // for(int n=0; n<N; n++){
            //     C[m*N+n] += val[idx]*B[k*N+n]; 
            // }
            // cblas_saxpy(n, val[idx], B+(k*N), 1, C+(m*N), 1);
            num_vector_neon(B+(k*N), val+idx, C+(m*N), n);
        }      
    }

}
void rowWiseBlocking(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){

    // cout << 1;
    int bsize = 8;
    // int R=0,Col=0;
    for(int R=0; R<M; R+=bsize){
        for(int Col=0; Col<N; Col+=bsize){
            #pragma omp parallel for num_threads(threads)
            for(int m=R; m<R+bsize; m++){
                int row_start = rowPtr[m];
                int row_end = rowPtr[m+1];
                for(int idx=row_start; idx<row_end; idx++){
                    int k=colInd[idx];
            
                    num_vector_neon(B+(k*N)+Col, val+idx, C+(m*N)+Col, bsize);
                }
            }
        }
    }

}
void rowWiseRegisterBlocking(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){
    int bsize=4;
    int n = N;
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        for(int kk=0; kk<n; kk+=bsize){
            float32x4_t acc;
            // acc = vmovq_n_f32(0);
            acc = vld1q_f32(C+m*n+kk);
            for(int idx=row_start; idx<row_end; idx++){
                int k=colInd[idx];
                float32x4_t a;
                float32_t b=*(val+idx);
                float32x4_t c;
                a = vld1q_f32(B+k*N+kk);
                
                acc = vfmaq_n_f32(acc, a, b);
                
            }
            vst1q_f32(C+m*n+kk, acc);
        }         
    }
}
void irregular(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){
    int n=N;
    int bsize=32;
    for(int m=0;m<M;m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        for(int idx=row_start; idx<row_end; ++idx){
            int k=colInd[idx];
            int Cols = N/bsize;
            #pragma omp parallel for num_threads(threads)
            for(int Col=0; Col<Cols; ++Col){
                // cout << omp_get_num_threads() << "--" << omp_get_thread_num() <<endl;
                int add = Col*bsize;
                num_vector_neon(B+(k*N)+add, val+idx, C+(m*N)+add, bsize);
                
            }

        }
    }
}
int main(int argc, char *argv[]){
    int M = strtol(argv[1], NULL, 10);
    int K = strtol(argv[2], NULL, 10);
    int N = strtol(argv[3], NULL, 10);
    int threads = strtol(argv[4], NULL, 10);
    int ite = 1;
    // int M=1000,K=1000,N=1000;
    float ratio = 0.1;
    int nozero;
    float* matrix = (float*)malloc(M*K*sizeof(float));
    float *B = (float *)malloc(K*N*sizeof(float));
    for(int i=0; i<M*K; ++i){matrix[i] = 0;}
    for(int i=0; i<N*K; ++i){B[i] = 0;}
    nozero = prepareSparse(matrix, M, K, ratio);
    // int* rowPtr = (int *)malloc(nozero*sizeof(int));
    // int* colInd = (int *)malloc(nozero*sizeof(int));
    // float* val =  (float *)malloc(nozero*sizeof(float));
    int* rowPtr2 = (int *)malloc((M+1)*sizeof(int));
    int* colInd2 = (int *)malloc(nozero*sizeof(int));
    float* val2 =  (float *)malloc(nozero*sizeof(float));
    int* rowPtr = (int *)malloc((M+1)*sizeof(int));
    int* colInd = (int *)malloc(nozero*sizeof(int));
    float* val =  (float *)malloc(nozero*sizeof(float));

    double toCSRTime;
    double timeStart = getHighResolutionTime();

    toCSR(matrix, M, K, rowPtr, colInd, val);
    double timeEnd = getHighResolutionTime();
    toCSRTime = timeEnd - timeStart;

    
    float *C = (float *)malloc(M*N*sizeof(float));
    float *C_compare = (float *)malloc(M*N*sizeof(float));
    float *C_compare2 = (float *)malloc(M*N*sizeof(float));
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            C[i*N+j]=0.f;
            C_compare[i*N+j]=0.f;
            C_compare2[i*N+j]=0.f;
        }
    }
    float num=1;
    for(int i=0;i<K;i++){
        for(int j=0;j<N;j++){
            B[i*N+j]=(float)(rand()%20000-10000)/1000;
        }        
    }
    // simple test
    // int M=8,K=7,N=6,threads = 2;
    // float ratio = 0.1;
    // int rowPtr[] = {0,4,7,9,14,15,21,25,27};
    // int colInd[] = {1,3,5,6,1,2,3,3,4,0,1,2,3,4,5,1,2,3,4,5,6,0,2,4,5,2,6};
    // float val[] =  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26};
    // float* B = (float *)malloc(K*N*sizeof(float));
    // float* C = (float *)malloc(M*N*sizeof(float));
    // for(int i=0;i<K;i++){
    //     for(int j=0;j<N;j++){
    //         B[i*N+j]=1;
    //     }
    // }
    // for(int i=0;i<M;i++){
    //     for(int j=0;j<N;j++){
    //         C[i*N+j]=0.f;
    //     }
    // }
    // show matrix
    // cout << "Matrix" << endl;
    // for(int i=0; i<M; i++){
    //     for(int j=0; j<K; j++){
    //         // cout << matrix[i*K+j] << " ";
    //         printf("%.1f ", matrix[i*K+j]);
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    // cout << "B" << endl;
    // for(int i=0; i<K; i++){
    //     for(int j=0; j<N; j++){
            
    //         printf("%.1f ", B[i*N+j]);
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // prepareSparse2(matrix, M, K, rowPtr2, colInd2, val2, ratio);
    // show Sparse
    // for(int i=0;i<nozero;i++) cout << rowPtr2[i] << " ";
    // cout << endl;
    // for(int i=0;i<nozero;i++) cout << colInd2[i] << " ";
    // cout << endl;
    // for(int i=0;i<nozero;i++) cout << val2[i] << " ";
    // cout << endl;
    // cout << 1 << endl;
    rowBaseSpmm3(rowPtr, colInd, val, B, C, M, K, N, threads);
    double t1, t2, take=0;
    t1 = getHighResolutionTime();
    for(int i=0; i<ite; i++){
        // clear(C, M, N);
        
        rowBaseSpmm3(rowPtr, colInd, val, B, C, M, K, N, threads);
        
    }
    t2 = getHighResolutionTime();
    take = t2-t1;
    

    double blas_timeTake=0;
    rowBaseSpmm2(rowPtr, colInd, val, B, C_compare, M, K, N, threads);
    t1 = getHighResolutionTime();
    for(int i=0; i<ite; i++){
        // clear(C_compare, M, N);
        
        rowBaseSpmm2(rowPtr, colInd, val, B, C_compare, M, K, N, threads);
    }
    t2 = getHighResolutionTime();
    blas_timeTake = t2-t1;
//    bool same = compare(M,N,C,C_compare);
//    if(same){
//        cout << "correct result!" << endl;
//    }else{
//        cout << "false result!" << endl;
//    }
    cout << "toCSRTime:" << toCSRTime << endl;
    cout <<"m k n ratio:" << M<<" "<< K <<" "<< N <<" "<<ratio << "  Time take:" << take/ite << endl;
    cout << "blas_timeTake:" << blas_timeTake/ite << endl;
    // show result C
    // cout << endl;
    // cout << "C[]" << endl;
    // for(int i=0;i<M;i++){
    //     for(int j=0;j<N;j++){
    //         std::cout << C[i*N+j] << " ";
    //     }
    //     std::cout <<std::endl;
    // }
    // cout << endl;
    // cout << "C_compare[]" << endl;
    // for(int i=0;i<M;i++){
    //     for(int j=0;j<N;j++){
    //         std::cout << C_compare[i*N+j] << " ";
    //     }
    //     std::cout <<std::endl;
    // }
    return 0;
}

int prepareSparse(float* matrix, int rows, int cols, float ratio){
    int nozero = 0;
    srand((unsigned)time(NULL));
    for(int h=0;h<ratio*rows*cols;h++) {
        int i=rand()%rows;
        int j=rand()%cols;
        matrix[i*cols + j] = (float)(rand()%20000-10000)/1000;
        nozero++;
        
    }
    return nozero;
    // toCSR(matrix, rows, cols, rowPtr, colInd, val);
    
}
void prepareSparse2(float* matrix, int rows, int cols,int* rowPtr, int* colInd, float* val, float ratio){

    srand((unsigned)time(NULL));
    for(int h=0;h<ratio*rows*cols;h++) {
      int i=rand()%rows;
      int j=rand()%cols;
      matrix[i*cols + j] = (float)(rand()%20000-10000)/1000;
    }
    
    toCSR2(matrix, rows, cols, rowPtr, colInd, val);
    
}
void toCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    int size = 0;
    for(int row=0;row < rows;row++){
        rowPtr[row] = size;
        for(int col=0;col < cols; col++){
            float num=matrix[row*cols + col];
            
            if(num != 0){
                val[size]=num;
                colInd[size] = col;
                size++;
            }
        }
    }
    rowPtr[rows] = size;
}
void toCSR2(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    int size = 0;
    for(int row=0;row < rows;row++){
        rowPtr[row] = size;
        int col,dir;
        if(row%2==0){
            col = 0;
            dir = 1;
        }
        else{
            col = cols-1;
            dir = -1;
        }
        while(col < cols && col >=0){
            float num=matrix[row*cols + col];
            
            if(num != 0){
                val[size]=num;
                colInd[size] = col;
                size++;
            }
            col+=dir;
        }
    }
    rowPtr[rows] = size;
}
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}
void mul(int m, int k, int n, float* A, float* B, float* C){
    for(int i=0; i<m; i++){
        for(int j=0;j<n;j++){
            for(int index=0; index<k; index++){
                // C[i][j] += A[i][index] * B[index][j];
                C[i*n+j] += A[i*k+index] * B[index*n + j];
            }
        }
    }
}
bool compare(int m,int n, float* C1, float* C2){
    bool same = true;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(abs(C1[i*n+j] - C2[i*n+j])>0.1) return false;
        }
    }
    return same;
}
void origin(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){
    omp_set_num_threads(threads);
    // int threads = 8;
    // #pragma omp parallel for num_threads(threads) collapse(2)
    // omp_set_nested(8);
    // #pragma omp parallel for num_threads(threads)
    // #pragma omp parallel
    // #pragma omp single
    // #pragma omp taskloop num_tasks (threads)
    #pragma omp parallel for num_threads(threads)
    for(int i=0;i<M;i++){
        
        
        for(int j=0; j<N;j++){
            int row_start = rowPtr[i];
            int row_end = rowPtr[i+1];
            float res = 0;
            for(int ptr=row_start; ptr < row_end; ptr++){
                int t = colInd[ptr];
                res += val[ptr] * B[t*N+j];
            }
            C[i*N+j] = res;
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
void clear(float* matrix, int m, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            matrix[i*n+j]=0;
        }
    }
}
