//g++ -g -O3 numaAwareSpMV.cpp -o numaAwareSpMV -I /home/nscc-gz/zhengj/opt/OpenBLAS/include -L /home/nscc-gz/zhengj/opt/OpenBLAS/lib -lpthread -lnuma -fopenmp
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <arm_neon.h>
#include <numa.h>
#include <math.h>
#define NUM_NODES 4
using namespace std;
typedef struct {
    int *rowPtr;
    int *colInd;
    float *val;
    float *matrix;
    float *B;
    float *C;
    int currentNode;
    int M;
    int K;
    int threadsPerNode;
}threadArguments;
void sparse_dense_vector2(float *B, float *C, int rowPtr[], int colInd[], float val[], int M,int threads);
double getHighResolutionTime(void);
//void vector_vector_neon(float32_t *A, float32_t *B, float32_t *C);
void sparse_dense_vector(float *B, float *C, int rowPtr[], int colInd[], float val[], int M, int threads);
void displaySparse(int *rowPtr, int *colInd, float *val, int M){
    for(int i=0; i<=M; i++) cout << rowPtr[i] << " ";
    cout << endl;
    for (int i = 0; i < rowPtr[M]; ++i) {
        printf("%d ", colInd[i]);
    }
    cout <<endl;
    for (int i = 0; i < rowPtr[M] ;++i) {
        printf("%.0f ", val[i]);
    }
    cout <<endl;
}
void toVectorAwareCSR(float* matrix,int rows, int cols, int* rowPtr, int* colInd, float* val){
    int size = 0;
    for(int row=0;row < rows;row++){
        rowPtr[row] = size;
        for(int col=0;col < cols; col+=4){
            bool load = false;
            for(int t=col; t<col+4 && t<cols; t++){
                if(fabs(matrix[row*cols+t]) > 0.1){
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
void clearMatrix(float *A, int M, int N){
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            A[i*N+j]=0;
        }
    }
}
void nnzCount(float* A, int M, int N, int nnzs[]){
    for(int t=0; t<4; t++){
        for(int i=t*M/4; i<(t+1)*M/4; i++){
            for (int j = 0; j < N ; ++j) {
                if(fabs(A[i*N+j])>0.1) nnzs[t]++;
            }
        }
    }
    return;
}
void displayDense(float* A, int M, int N){
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%lf ", A[i*N+j]);
        }
        printf("\n");
    }
}

void prepareSparse(float* matrix, int rows, int cols, float ratio){
    int nozero = ratio*rows*cols;
    srand((unsigned)time(NULL));
    for(int h=0;h<nozero;h++) {
        int i=rand()%rows;
        int j=rand()%cols;

        int num = (int)(rand()%10);
        matrix[i*cols + j] = (float )num;
    }
}
void distributeSparse(float* A, int M, int N, float** AData){
    int blockSize = M/4*N;
    for(int node=0; node<NUM_NODES; node++){
        memcpy(AData[node], A+node*blockSize, sizeof(float )*blockSize);
    }
}
void* ThreadWork(void *args){
    threadArguments *a = (threadArguments*)args;
    numa_run_on_node(a->currentNode);
    int M = a->M, K = a->K;
    int* rowPtr = a->rowPtr;
    int* colInd = a->colInd;
    float* val = a->val;
    float* B = a->B;
    float* C = a->C;
    int threads = a->threadsPerNode;
    float *matrix = a->matrix;

    double start = getHighResolutionTime();
    sparse_dense_vector(B, C, rowPtr, colInd, val, M, threads);
    double timetake = getHighResolutionTime()-start;
    printf("%d time %lf\n", a->currentNode, timetake);
    pthread_exit((void*) args);
}
int main(int argc, char *argv[]){
    int M = strtol(argv[1], NULL, 10);
    int K = strtol(argv[2], NULL, 10);
    int threads = strtol(argv[3], NULL, 10);

    int ite = 1;
    float ratio = 0.6;
    int nozero = (int)(ratio*M*K);
    float* matrix = (float*)malloc(M*K*sizeof(float));
    int nnzs[4]={0,0,0,0};
    clearMatrix(matrix, M, K);
    prepareSparse(matrix, M, K, ratio);
    nnzCount(matrix, M, K, nnzs);
    float** AData = (float**)malloc(NUM_NODES*sizeof(float* ));
    int** rowPtrData = (int**)malloc(NUM_NODES*sizeof(int*));
    int** colIndData = (int**)malloc(NUM_NODES*sizeof(int*));
    float** valData = (float**)malloc(NUM_NODES*sizeof(float*));
    float** BData = (float**)malloc(NUM_NODES*sizeof(float*));
    float** CData = (float**)malloc(NUM_NODES*sizeof(float*));
    for(int node=0; node<NUM_NODES; ++node){
        AData[node] = (float *)malloc(M/4*K*sizeof(float ));
        rowPtrData[node] = (int*)numa_alloc_onnode((M/4+1)*sizeof(int), node);
        colIndData[node] = (int*)numa_alloc_onnode(4*nnzs[node]*sizeof(int), node);
        valData[node] = (float *)numa_alloc_onnode(4*nnzs[node]*sizeof(float), node);
        BData[node] = (float*)numa_alloc_onnode(K*sizeof(float), node);
        CData[node] = (float*)numa_alloc_onnode(M/4*sizeof(float), node);

    }

    distributeSparse(matrix, M, K, AData);

#pragma omp parallel for num_threads(NUM_NODES)
    for (int i = 0; i < NUM_NODES; ++i) {
        toVectorAwareCSR(AData[i], M/4, K, rowPtrData[i], colIndData[i], valData[i]);
    }

#pragma omp parallel for num_threads(NUM_NODES)
    for (int i = 0; i < NUM_NODES; ++i) {
        for(int j=0; j<K; j++){
            BData[i][j]=5;
        }
    }
#pragma omp parallel for num_threads(NUM_NODES)
    for (int i = 0; i < NUM_NODES; ++i) {
        for(int j=0; j<M/4; j++){
            CData[i][j]=0;
        }
    }
    double s=getHighResolutionTime();
    printf("%d %d %d %d\n", nnzs[0], nnzs[1], nnzs[2], nnzs[3]);
    printf("%d\n", rowPtrData[0][M/4]);
    sparse_dense_vector(BData[0], CData[0], rowPtrData[0], colIndData[0], valData[0], M/4, threads);
    double time1 = getHighResolutionTime()-s;
    printf("%lf\n", time1);
    int rc;
    pthread_t thread[NUM_NODES];
    threadArguments threadArgs[NUM_NODES];
    pthread_attr_t attr;
    void *status;
    printf("Running parallelSum------------------\n");
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for(int i=0; i<NUM_NODES; ++i){
        int currentNode = i;
        threadArgs[i].currentNode = currentNode;
        threadArgs[i].rowPtr = rowPtrData[currentNode];
        threadArgs[i].colInd = colIndData[currentNode];
        threadArgs[i].val = valData[currentNode];
        threadArgs[i].B = BData[currentNode];
        threadArgs[i].C = CData[currentNode];
        threadArgs[i].M = M/4;
        threadArgs[i].K = K;
        threadArgs[i].threadsPerNode = threads/NUM_NODES;
    }
    double start = getHighResolutionTime();
    for(int i=0; i<NUM_NODES; ++i){
        rc = pthread_create(&thread[i], &attr, ThreadWork, (void*)&threadArgs[i]);
        if(rc) {
            fprintf(stderr,"Error - pthread_create() return code: %d\n",rc);
            exit(EXIT_FAILURE);
        }

    }
    pthread_attr_destroy(&attr);
    double mid = getHighResolutionTime();
    for (int i = 0; i < NUM_NODES; i++) {
        rc = pthread_join(thread[i], &status);
//        if (rc) {
//            printf("ERROR; return code from pthread_join() is %d\n", rc);
//            exit(EXIT_FAILURE);
//        }
    }
    double end = getHighResolutionTime();

    printf("ratio %lf, TimeTake %lf s\n",ratio, end-start);
    printf("time1 %lf\n", mid - start);
    printf("time2 %lf\n", end-mid);

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
void sparse_dense_vector2(float *B, float *C, int rowPtr[], int colInd[], float val[], int M,int threads){

    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];

        float sum=0;
        for(int idx=row_start; idx<row_end; idx+=4){
            int k=colInd[idx];

            sum+= val[idx]*B[k];
        }
        C[m]+=sum;

    }
}
void sparse_dense_vector(float *B, float *C, int rowPtr[], int colInd[], float val[], int M,int threads){

    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        if(m==0) printf("%d\n", omp_get_num_threads());
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];

        float32x4_t sum_vec=vdupq_n_f32(0),left_vec,right_vec;
        for(int idx=row_start; idx<row_end; idx+=4){
            int k=colInd[idx];

            left_vec = vld1q_f32(val+idx);//src_reg1 = dataload(val+idx)
            right_vec=vld1q_f32(B+k); //src_reg2 = dataload(B+k)
            sum_vec=vmlaq_f32(sum_vec,left_vec,right_vec);//des_reg = vmlaq_f32(src_reg1, src_reg2)

        }
        float32x2_t r=vadd_f32(vget_high_f32(sum_vec),vget_low_f32(sum_vec));//
        C[m]+=vget_lane_f32(vpadd_f32(r,r),0);

    }
}
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}