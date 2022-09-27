//g++ -g -O3 SpMM.cpp -o SpMM -I /home/nscc-gz/zhengj/opt/OpenBLAS/include -L /home/nscc-gz/zhengj/opt/OpenBLAS/lib -lopenblas -fopenmp
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>
#include <arm_neon.h>
// #include "rowBaseSpmm.h"
using namespace std;
double getHighResolutionTime(void);
void CSRToMatrix(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val);
void sparse_matrix_mul3(int rowPtr1[], int colInd1[], float val1[],
                int rowPtr2[], int colInd2[], float val2[], 
                float* matrixC, int M, int K, int N, int threads);
void sparse_matrix_mul2(int rowPtr1[], int colInd1[], float val1[],
                int rowPtr2[], int colInd2[], float val2[], 
                int rowPtr3[], int colInd3[], float val3[],int M, int K, int N, int threads,float* acc);
void sparse_matrix_mul(int rowPtr1[], int colInd1[], float val1[],
                int colPtr2[], int rowInd2[], float val2[], 
                int rowPtr3[], int colInd3[], float val3[],int M,int K,int N, int threads);
void vector_vector_neon(float32_t *A, float32_t *B, float32_t *C);
void sparse_dense_vector(float *B, float *C, int rowPtr[], int colInd[], float val[], int M, int threads);
void toVectorAwareCSR(float* matrix,int rows, int cols, int* rowPtr, int* colInd, float* val){
    int size = 0;
    for(int row=0;row < rows;row++){
        rowPtr[row] = size;
        for(int col=0;col < cols; col+=4){
            bool load = false;
            for(int t=col; t<col+4 && t<cols; t++){
                if(matrix[row*cols+t] != 0){
                    load = true;
                    break;
                }
            }
            if(load){
                for(int t=col; t<col+4; t++){
                    val[size]=matrix[row*cols + t];
                    colInd[size] = t;
                    size++;
                }
            }
            
        }
    }
    rowPtr[rows] = size;
}
void prepareSparse(float* matrix, int rows, int cols, float ratio){
    int nozero = ratio*rows*cols;
    srand((unsigned)time(NULL));
    for(int h=0;h<nozero;h++) {
      int i=rand()%rows;
      int j=rand()%cols;
      if(matrix[i*cols + j] != 0){
          h--;
          continue;
      }
      matrix[i*cols + j] = (float)(rand()%20000-10000)/1000;
    }
    
    // toCSR(matrix, rows, cols, rowPtr, colInd, val);
    
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
void toCSC(float* matrix, int rows,int cols, int* colPtr, int* rowInd, float* val){
    int size = 0;
    for(int col=0; col < cols;col++){
        colPtr[col] = size;
        for(int row=0;row < rows; row++){
            float num=matrix[row*cols + col];
            
            if(num != 0){
                val[size]=num;
                rowInd[size] = row;
                size++;
            }
        }
    }
    colPtr[cols] = size;
}
void CSRToMatrix(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    for(int i=0; i<rows; i++){
        int row_start = rowPtr[i];
        int row_end = rowPtr[i+1];
        for(int idx=row_start; idx<row_end; idx++){
            matrix[i*cols+colInd[idx]] = val[idx];
        }
    }
}
int main(int argc, char *argv[]){
    // simple test
    
    int M = strtol(argv[1], NULL, 10);
    int K = strtol(argv[2], NULL, 10);
    int N = strtol(argv[3], NULL, 10);
    int threads = strtol(argv[4], NULL, 10);
    int ite = 30;
    // int M=1000,K=1000,N=1000;
    
    float ratio = 0.01;
    int nozero = (int)(ratio*M*K);
    int cap = (int)1.2*nozero;
    float* matrix = (float*)malloc(M*K*sizeof(float));
    float* matrixB = (float*)malloc(N*K*sizeof(float));
    int* rowPtr1 = (int *)malloc((M+1)*sizeof(int));
    int* colInd1 = (int *)malloc(nozero*sizeof(int));
    float* val1 =  (float *)malloc(nozero*sizeof(float));
    int* rowPtr2 = (int *)malloc((K+1)*sizeof(int));
    int* colInd2 = (int *)malloc(nozero*sizeof(int));
    float* val2 =  (float *)malloc(nozero*sizeof(float));
    int* rowPtr3 = (int *)malloc((M+1)*sizeof(int));
    int* colInd3 = (int *)malloc(2*nozero*sizeof(int));
    float* val3 =  (float *)malloc(2*nozero*sizeof(float));
    
    float* acc = (float*)malloc(threads*N*sizeof(float));
    
    // float* C = (float*)malloc(M*sizeof(float));
    // float* B = (float*)malloc(K*sizeof(float));
    // // for(int i=0; i<K; i++){
    // //     B[i]=2;
    // // }
    // for(int i=0; i<M; i++){
    //     C[i]=0;
    // }
    prepareSparse(matrix, M, K, ratio);
    prepareSparse(matrixB, K, N, ratio);
    toCSR(matrix, M, K, rowPtr1, colInd1, val1);
    toCSR(matrixB, K, N, rowPtr2, colInd2, val2);
    float* matrixC = (float*)malloc(M*N*sizeof(float));

    double start,timeElapse;
    start = getHighResolutionTime();
    for(int i=0; i< ite; i++){
        sparse_matrix_mul3(rowPtr1, colInd1, val1,
                rowPtr2, colInd2, val2, 
                matrixC,M,K,N, threads);
    }
    timeElapse = getHighResolutionTime() - start;
    cout << "M k N sparity: " << M << " " << K << " " << N << " " << ratio << endl;
    cout << "timeTake: " << timeElapse/ite << endl;
    
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
    // cout << "MatrixB" << endl;
    // for(int i=0; i<M; i++){
    //     for(int j=0; j<K; j++){
    //         // cout << matrix[i*K+j] << " ";
    //         printf("%.1f ", matrixB[i*K+j]);
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    // // show B
    // cout << "B" << endl;
    // cout << "nozero:" << nozero << endl;
    // for(int i=0;i<K+1;i++) cout << rowPtr2[i] << " ";
    // cout << endl;
    // for(int i=0;i<nozero;i++) cout << colInd2[i] << " ";
    // cout << endl;
    // for(int i=0;i<nozero;i++) cout << val2[i] << " ";
    // cout << endl;

    // cout << "C" << endl;
    // for(int i=0;i<M+1;i++) cout << rowPtr3[i] << " ";
    // cout << endl;
    // for(int i=0;i<cap;i++) cout << colInd3[i] << " ";
    // cout << endl;
    // for(int i=0;i<cap;i++) cout << val3[i] << " ";
    // cout << endl;

    // float* matrixC = (float*)malloc(M*N*sizeof(float));
    // for(int i=0; i< M; i++){
    //     for(int j = 0; j< N; j++){
    //         matrixC[i*N+j] = 0;
    //     }
    // }
    // CSRToMatrix(matrixC, M, N, rowPtr3, colInd3, val3);
    // cout << "Matrix C" << endl;
    // for(int i=0; i<M; i++){
    //     for(int j=0; j<N; j++){
    //         // cout << matrix[i*K+j] << " ";
    //         printf("%.1f ", matrixC[i*N+j]);
    //     }
    //     cout << endl;
    // }
    
    
}
void vector_vector_neon(float32_t *A, float32_t *B, float32_t *C){
    // dot mul
    float32x4_t a;
    float32x4_t b;
    float32x4_t c;
    a = vld1q_f32(A);
    c = vld1q_f32(C);
    b = vld1q_f32(B);
    c = vfmaq_laneq_f32(c, a, b, 0);
    vst1q_f32(C, c);
}
float dot(float* A,float* B,int K)
{
    float sum=0;
    float32x4_t sum_vec=vdupq_n_f32(0),left_vec,right_vec;
    for(int k=0;k<K;k+=4)
    {
        left_vec=vld1q_f32(A+ k);
        right_vec=vld1q_f32(B+ k);
        sum_vec=vmlaq_f32(sum_vec,left_vec,right_vec);
    }

    float32x2_t r=vadd_f32(vget_high_f32(sum_vec),vget_low_f32(sum_vec));
    sum+=vget_lane_f32(vpadd_f32(r,r),0);

    return sum;
}
void sparse_matrix_mul(int rowPtr1[], int colInd1[], float val1[],
                int colPtr2[], int rowInd2[], float val2[], 
                int rowPtr3[], int colInd3[], float val3[],int M, int K, int N, int threads){ 

    int size = 0;
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr1[m];
        int row_end = rowPtr1[m+1];
        rowPtr3[m] = size;
        for(int n=0; n<N; n++){
            int col_start = colPtr2[n];
            int col_end = colPtr2[n+1];
            int acc=0;
            
            for(int le=row_start, ri=col_start; le<row_end && ri<col_end; ){
                if(colInd1[le] == rowInd2[ri]){
                    acc += val1[le]*val2[ri];
                    le++;ri++;
                }else{
                    le < ri ? le++:ri++;
                }
            }

            if(acc!=0){
                val3[size]=acc;
                colInd3[size]=n;
                size++;
            }
        }
    }
    rowPtr3[M]=size;
}
void sparse_matrix_mul2(int rowPtr1[], int colInd1[], float val1[],
                int rowPtr2[], int colInd2[], float val2[], 
                int rowPtr3[], int colInd3[], float val3[], int M, int K, int N, int threads,float* acc){ 

    int size = 0;
    
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        // cout << omp_get_num_threads() << endl;
        int id = omp_get_thread_num();
        
        float* buffer = acc+id*N;
        int row_start1 = rowPtr1[m];
        int row_end1 = rowPtr1[m+1];
        bool* notzero = (bool*)malloc(N*sizeof(bool));

        for(int idx=row_start1; idx<row_end1; idx++){
            int k=colInd1[idx];
            for(int t=rowPtr2[k]; t<rowPtr2[k+1]; t++){
                buffer[colInd2[t]] += val1[idx]*val2[t];
                notzero[colInd2[t]] = true; 
            }
        }
        rowPtr3[m] = size;
        for(int i=0;i<N; i++){
            if(notzero){
                val3[size]=buffer[i];
                buffer[i]=0;//
                colInd3[size] = i;
                size++;
            }
        }
        // free(acc);
    }
    
    rowPtr3[M]=size;
}
void sparse_matrix_mul3(int rowPtr1[], int colInd1[], float val1[],
                int rowPtr2[], int colInd2[], float val2[], 
                float* matrixC, int M, int K, int N, int threads){ 

    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        
        int row_start1 = rowPtr1[m];
        int row_end1 = rowPtr1[m+1];
        for(int idx=row_start1; idx<row_end1; idx++){
            int k=colInd1[idx];
            for(int t=rowPtr2[k]; t<rowPtr2[k+1]; t++){
                matrixC[m*N+colInd2[t]] += val1[idx]*val2[t];
            }
        }
        
    }
    
}
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}