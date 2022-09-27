//g++ -g -O3 numaAwareSPDM.cpp -o numaAwareSPDM -I /home/nscc-gz/zhengj/opt/OpenBLAS/include -L /home/nscc-gz/zhengj/opt/OpenBLAS/lib -lopenblas -lpthread -lnuma
//test
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <pthread.h>
#include <cblas.h>
#include <string.h>
//#include <arm_neon.h>
#include <numa.h>
#include <stdlib.h>

#define THREAD_PERNODE 8
#define BLOCK_WITH (4096/THREAD_PERNODE)
#define NUM_NODES 4
#define NUM_THREADS 32

using namespace std;

typedef struct {
    int *colPtr;
    int *rowInd;
    float *val;
    float *B;
    float *C;
    int currentNode;
    int NodethreadID;
    int M;
    int K;
    int N;
}threadArguments;

double getHighResolutionTime(void) {
    struct timeval tod;
    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}
void toCSC(float* matrix, int M,int K, int* colPtr, int* rowInd, float* val){
    int size = 0;
    for(int col=0; col < K;col++){
        colPtr[col] = size;
        for(int row=0;row < M; row++){
            float num=matrix[row*K + col];

            if(num != 0){
                val[size]=num;
                rowInd[size] = row;
                size++;
            }
        }
    }
    colPtr[K] = size;
}
void sparseDistribute(float* matrix, int M,int K,
                      int** colPtrData, int** rowIndData, float** valData){
    for(int node=0; node<NUM_NODES; ++node){
        int size = 0;
        for(int col=node*K/NUM_NODES; col<(node+1)*K/NUM_NODES; ++col){
            colPtrData[node][col-node*K/NUM_NODES] = size;
            for(int row=0; row<M; ++row){
                float num = matrix[row*K+col];
                if(num!=0){
                    valData[node][size] = num;
                    rowIndData[node][size] = row;
                    size++;
                }
            }
        }
        colPtrData[node][K/NUM_NODES]=size;
    }
}
void copy(float* des, float* sour, int len){
    for(int i=0; i<len; ++i){
        des[i] = sour[i];
    }
    return;
}
void denseDistribute(float* B, float** BData, int K, int N){

    int partitionSize = (K/4)*N;
    for(int i=0; i<NUM_NODES; ++i){
        memcpy(BData[i], B+i*partitionSize, partitionSize*sizeof(float ));
//        copy(BData[i], B+i*partitionSize, partitionSize);
    }
    return;
}
void displayDense(float* A, int M, int N){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; ++j){
            printf("%6.1f\t", A[i*N+j]);
        }
        cout << endl;
    }
}
void displaySparse(int *colPtr, int *rowInd, float *val, int M, int K, int nnz){
    for(int i=0; i<=K; i++) cout << colPtr[i] << " ";
    cout << endl;
    for (int i = 0; i < nnz; ++i) {
        printf("%d ", rowInd[i]);
    }
    cout <<endl;
    for (int i = 0; i < nnz; ++i) {
        printf("%.1f ", val[i]);
    }
    cout <<endl;
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
//        matrix[i*cols + j] = (float)(rand()%20000-10000)/1000;
        matrix[i*cols + j] = (float)(rand()%10-0);
    }

    // toCSR(matrix, rows, cols, rowPtr, colInd, val);

}

void nnzCount(float* A, int M, int N, int nnzs[]){
    for(int t=0; t<4; t++){
        for(int i=0; i<M; i++){
            for (int j = t*N/4; j < (t+1)*N/4 ; ++j) {
                if(A[i*N+j]!=0) nnzs[t]++;
            }
        }
    }
    return;
}
void clearMatrix(float *A, int M, int N){
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            A[i*N+j]=0;
        }
    }
}
void numaAwareSPDM(int colPtr[], int rowInd[], float val[], float matrixB[], float matrixC, int M, int N, int K){
//    if(numa_available() < 0) return;
//    int nodes = numa_num_configured_nodes();
    int nodes = 4;
    int threadsPerNode = 4;
    float* sparseTask[4] = {};
}
void saxpy(int len, int num, float* vec, float *res){
    for(int i=0; i<len; ++i){
        res[i] += vec[i]*num;
    }
}
void NodeCalculateCore(int *colPtr, int *rowInd, float *val, float *B, float *C, int M, int K, int N, int block_len){
    for(int j=0; j<K; ++j){
        int col_start = colPtr[j];
        int col_end = colPtr[j+1];
        for(int idx=col_start; idx<col_end; ++idx){
            int row=rowInd[idx];//sparse element on (row,j)
            cblas_saxpy(block_len, val[idx], B+j*N, 1, C+row*N, 1);
//            saxpy(block_len, val[idx], B+j*N, C+row*N);
        }

    }
}

void* ThreadWork(void *args){
    threadArguments *a = (threadArguments*)args;
    numa_run_on_node(a->currentNode);
    int M = a->M, K = a->K, N = a->N;
    int NodethreadID = a->NodethreadID;
    int* colPtr = a->colPtr;
    int* rowInd = a->rowInd;
    float* val = a->val;
    float* B = a->B;
    float* C = a->C;

    int block_len = N/THREAD_PERNODE;

    float *BlockPtr = B+NodethreadID*block_len;
    float *CPtr = C+NodethreadID*block_len;
    NodeCalculateCore(colPtr, rowInd, val, BlockPtr, CPtr, M, K, N, block_len);

    pthread_exit((void*) args);
}


void sumUp(float** CData, int M, int N){
    for(int node=1; node<NUM_NODES; ++node){
        for(int i=0; i<M; ++i){
            for(int j=0; j<N; ++j){
                CData[0][i*N+j] += CData[node][i*N+j];
            }
        }
    }
}
bool compare(int m,int n, float* C1, float* C2){
    bool same = true;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float res = C1[i*n+j] - C2[i*n+j];
            if(res>0.1 or res < -0.1) return false;
        }
    }
    return same;
}

void naiveSPMM(float* A, float *B, float *C, int M, int K, int N){
    for(int i = 0; i < M; ++i)//矩阵相乘
    {
        for(int j=0; j<N; ++j){
            for(int t=0; t<K; ++t){
                C[i*N+j] += A[i*K+t] * B[t*N+j];
            }
        }
    }

}
bool validate(float *A, float *B, int M, int K, int N, float** CData){
    float *C = (float*)malloc(M*N*sizeof(float ));
    clearMatrix(C, M, N);
    naiveSPMM(A, B, C, M, K, N);
    if(compare(M, N, C, CData[0])){
        printf("Correct Result\n");
    }else{
        printf("False Result\n");
//        displayDense(C, M, N);
//        printf("myresult:\n");
//        displayDense(CData[0], M, N);
    }
}
int main(int argc, char *argv[]){
    int M = strtol(argv[1], NULL, 10);
    int K = strtol(argv[2], NULL, 10);
    int N = strtol(argv[3], NULL, 10);
//    int M = 1024, K=1024, N=1024;
    int ite = 1;
    float ratio = 0.1;
    int nnz = (int)(ratio*M*K);
    float* matrix = (float*)numa_alloc_onnode(M*K*sizeof(float), 0);
    float* B = (float*)numa_alloc_onnode(N*K*sizeof(float), 0);
//    float* matrix = (float*)malloc(M*K*sizeof(float));
//    float* B = (float*)malloc(N*K*sizeof(float));
    clearMatrix(matrix, M, K);
    prepareSparse(matrix, M, K, ratio);
    for(int i=0;i<K;i++){
        for(int j=0;j<N;j++){
//            B[i*N+j]=(float)(rand()%20000-10000)/1000;
            B[i*N+j]=(float)(rand()%10-0);
        }
    }
    int nnzs[4]={0,0,0,0};
    nnzCount(matrix, M, K, nnzs);

    int** colPtrData = (int**)malloc(NUM_NODES*sizeof(int*));
    int** rowIndData = (int**)malloc(NUM_NODES*sizeof(int*));
    float** valData = (float**)malloc(NUM_NODES*sizeof(float*));
    float** BData = (float**)malloc(NUM_NODES*sizeof(float*));
    float** CData = (float**)malloc(NUM_NODES*sizeof(float*));
    for(int node=0; node<NUM_NODES; ++node){
        colPtrData[node] = (int*)numa_alloc_onnode((K/4+1)*sizeof(int), node);
        rowIndData[node] = (int*)numa_alloc_onnode(nnzs[node]*sizeof(int), node);
        valData[node] = (float *)numa_alloc_onnode(nnzs[node]*sizeof(float), node);
        BData[node] = (float*)numa_alloc_onnode(K/4*N*sizeof(float), node);
        CData[node] = (float*)numa_alloc_onnode(M*N*sizeof(float), node);
//        colPtrData[node] = (int*)malloc((K/4+1)*sizeof(int));
//        rowIndData[node] = (int*)malloc(nnzs[node]*sizeof(int));
//        valData[node] = (float *)malloc(nnzs[node]*sizeof(float));
//        BData[node] = (float*)malloc(K/4*N*sizeof(float));
//        CData[node] = (float*)malloc(M*N*sizeof(float));
    }
    for(int i=0; i<NUM_NODES; ++i){
        clearMatrix(CData[i], M, N);
    }
    sparseDistribute(matrix, M, K, colPtrData, rowIndData, valData);
    double memcopyTime = 0, end=0;
    double start = getHighResolutionTime();
    denseDistribute(B, BData, K, N);
    memcopyTime = getHighResolutionTime()-start;


    int rc;
    pthread_t thread[NUM_THREADS];
    threadArguments threadArgs[NUM_THREADS];
    pthread_attr_t attr;
    void *status;


    printf("Running parallelSum------------------\n");
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for(int i=0; i<NUM_THREADS; ++i){
        int currentNode = NUM_NODES * i / NUM_THREADS;
        threadArgs[i].currentNode = currentNode;
        threadArgs[i].colPtr = colPtrData[currentNode];
        threadArgs[i].rowInd = rowIndData[currentNode];
        threadArgs[i].val = valData[currentNode];
        threadArgs[i].B = BData[currentNode];
        threadArgs[i].C = CData[currentNode];
        threadArgs[i].NodethreadID = i % THREAD_PERNODE;
        threadArgs[i].M = M;
        threadArgs[i].K = K/NUM_NODES;
        threadArgs[i].N = N;
    }
    start = getHighResolutionTime();
    for(int i=0; i<NUM_THREADS; ++i){

        rc = pthread_create(&thread[i], &attr, ThreadWork, (void*)&threadArgs[i]);
        if(rc) {
            fprintf(stderr,"Error - pthread_create() return code: %d\n",rc);
            exit(EXIT_FAILURE);
        }

    }

    /* Free attribute and wait for the other threads */
    pthread_attr_destroy(&attr);
    double mid = getHighResolutionTime();
    for (int i = 0; i < NUM_THREADS; i++) {
        rc = pthread_join(thread[i], &status);
//        if (rc) {
//            printf("ERROR; return code from pthread_join() is %d\n", rc);
//            exit(EXIT_FAILURE);
//        }
    }
    end = getHighResolutionTime();

//    printf("\n----------------\n");
//    for(int i=0; i<NUM_NODES; ++i){
//        displaySparse(colPtrData[i], rowIndData[i], valData[i], M, K/4, nnzs[i]);
//        printf("\n");
//    }
//    printf("\n----------------\n");
    printf("TimeTake %lf s\n", end-start);
    printf("time1 %lf\n", mid - start);
    printf("time2 %lf\n", end-mid);
//    printf("matrix\n");
//    displayDense(matrix, M, K);
//    printf("B:\n");
//    displayDense(B, K, N);
//    printf("C:\n");
//    displayDense(CData[0], M, N);
    double sumUpTime = 0;
    start= getHighResolutionTime();
    sumUp(CData, M, N);
    sumUpTime = getHighResolutionTime()-start;
    printf("--memCopyTime: %lf s   sumUpTime: %lf s\n", memcopyTime, sumUpTime);
//    validate(matrix, B, M, K, N, CData);
}