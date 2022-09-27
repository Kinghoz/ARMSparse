//g++ -g -O3 numaAwareSDMM.cpp -o numaAwareSDMM -I /home/nscc-gz/zhengj/opt/OpenBLAS/include -L /home/nscc-gz/zhengj/opt/OpenBLAS/lib -lopenblas -lpthread -lnuma -fopenmp
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include <pthread.h>
#include <numa.h>
//#include <cblas.h>
//#include <arm_neon.h>
// #include "rowBaseSpmm.h"
#define NUM_NODES 4
using namespace std;
typedef struct {
    int *ArowPtr;
    int *AcolInd;
    float *Aval;
    int *BrowPtr;
    int *BcolInd;
    float *Bval;
    float *C;
    int currentNode;
    int M;
    int K;
    int N;
    int threadPerNode;
}threadArguments;

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


void sumUp(float** CData, int M, int N, int threads){
    for(int node=1; node<NUM_NODES; ++node){
#pragma omp parallel for num_threads(threads)
        for(int i=0; i<M; ++i){
            for(int j=0; j<N; ++j){
                CData[0][i*N+j] += CData[node][i*N+j];
            }
        }
    }
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
      int t = (int)(rand()%10);
      matrix[i*cols + j] = (float)t;
    }
    
    // toCSR(matrix, rows, cols, rowPtr, colInd, val);
    
}
void displayDense(float* A, int M, int N){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; ++j){
            printf("%.1lf\t", A[i*N+j]);
        }
        cout << endl;
    }
}
void DistributetoCSR(float* matrix, int M,int N, int** rowPtrData, int** colIndData, float** valData){

    for (int node = 0; node < NUM_NODES; ++node) {
        int size = 0;
        for(int row=0; row<M; row++){
            rowPtrData[node][row] = size;
            for (int col = node*N/4; col < (node+1)*(N/4); ++col) {
                float num=matrix[row*N + col];
                if(fabs(num) > 0.1){
                    valData[node][size]=num;
                    colIndData[node][size] = col-node*N/4;
                    size++;
                }
            }
        }
        rowPtrData[node][M] = size;
    }
}
void toCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val){
    int size = 0;
    for(int row=0;row < rows;row++){
        rowPtr[row] = size;
        for(int col=0;col < cols; col++){
            float num=matrix[row*cols + col];
            
            if(fabs(num) > 0.1){
                val[size]=num;
                colInd[size] = col;
                size++;
            }
        }
    }
    rowPtr[rows] = size;
}
void displaySparse(int *rowPtr, int *colInd, float *val, int M, int K, int nnz){
    for(int i=0; i<=M; i++) cout << rowPtr[i] << " ";
    cout << endl;
    for (int i = 0; i < nnz; ++i) {
        printf("%d ", colInd[i]);
    }
    cout <<endl;
    for (int i = 0; i < nnz; ++i) {
        printf("%.0f ", val[i]);
    }
    cout <<endl;
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
void clearMatrix(float *A, int M, int N){
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            A[i*N+j]=0;
        }
    }
}
void colSplitnnzCount(float* A, int M, int N, int nnzs[]){
    for(int t=0; t<4; t++){
        for(int i=0; i<M; i++){
            for (int j = t*N/4; j < (t+1)*N/4 ; ++j) {
                if(fabs(A[i*N+j]-0) > 0.1) nnzs[t]++;
            }
        }
    }
    return;
}
void rowSplitnnzCount(float* A, int M, int N, int nnzs[]){
    for(int t=0; t<4; t++){
        for(int i=t*M/4; i<(t+1)*M/4; i++){
            for (int j = 0; j < N ; ++j) {
                if(fabs(A[i*N+j]-0) > 0.1) nnzs[t]++;
            }
        }
    }
    return;
}
void* ThreadWork(void *args){
    threadArguments *a = (threadArguments*)args;
    numa_run_on_node(a->currentNode);
    int M = a->M, K = a->K, N = a->N;
    int* ArowPtr = a->ArowPtr;
    int* AcolInd = a->AcolInd;
    float* Aval = a->Aval;
    int* BrowPtr = a->BrowPtr;
    int* BcolInd = a->BcolInd;
    float* Bval = a->Bval;
    float* C = a->C;
    int threadPerNode = a->threadPerNode;



    double start = getHighResolutionTime();
    sparse_matrix_mul3(ArowPtr, AcolInd, Aval,
                BrowPtr, BcolInd, Bval,
                C,M,K,N, threadPerNode);
    double timetake = getHighResolutionTime()-start;
//    printf("%d time %lf\n", a->currentNode, timetake);
    pthread_exit((void*) args);
}
int main(int argc, char *argv[]){
    // simple test
    
    int M = strtol(argv[1], NULL, 10);
    int K = strtol(argv[2], NULL, 10);
    int N = strtol(argv[3], NULL, 10);
    int threads = strtol(argv[4], NULL, 10);
    int ite = 30;
    // int M=1000,K=1000,N=1000;
    
    float ratio = 0.1;
    int nozero = (int)(ratio*M*K);
    int cap = (int)1.2*nozero;
    float* matrix = (float*)malloc(M*K*sizeof(float));
    float* matrixB = (float*)malloc(N*K*sizeof(float));
    prepareSparse(matrix, M, K, ratio);
    prepareSparse(matrixB, K, N, ratio);
//    int* rowPtr1 = (int *)malloc((M+1)*sizeof(int));
//    int* colInd1 = (int *)malloc(nozero*sizeof(int));
//    float* val1 =  (float *)malloc(nozero*sizeof(float));
//    int* rowPtr2 = (int *)malloc((K+1)*sizeof(int));
//    int* colInd2 = (int *)malloc(nozero*sizeof(int));
//    float* val2 =  (float *)malloc(nozero*sizeof(float));
    int Annz[4]={0,0,0,0};
    int Bnnz[4]={0,0,0,0};
    colSplitnnzCount(matrix, M, K, Annz);
    rowSplitnnzCount(matrixB, K, N, Bnnz);
    int** ArowPtrData = (int**)malloc(NUM_NODES*sizeof(int*));
    int** AcolIndData = (int**)malloc(NUM_NODES*sizeof(int*));
    float** AvalData = (float**)malloc(NUM_NODES*sizeof(float*));
    int** BrowPtrData = (int**)malloc(NUM_NODES*sizeof(int*));
    int** BcolIndData = (int**)malloc(NUM_NODES*sizeof(int*));
    float** BvalData = (float**)malloc(NUM_NODES*sizeof(float*));
    float** CData = (float**)malloc(NUM_NODES*sizeof(float*));
    for(int node=0; node<NUM_NODES; ++node){
        ArowPtrData[node] = (int*)numa_alloc_onnode((M+1)*sizeof(int), node);
        AcolIndData[node] = (int*)numa_alloc_onnode(Annz[node]*sizeof(int), node);
        AvalData[node] = (float *)numa_alloc_onnode(Bnnz[node]*sizeof(float), node);

        BrowPtrData[node] = (int*)numa_alloc_onnode((M+1)*sizeof(int), node);
        BcolIndData[node] = (int*)numa_alloc_onnode(Annz[node]*sizeof(int), node);
        BvalData[node] = (float *)numa_alloc_onnode(Bnnz[node]*sizeof(float), node);
        CData[node] = (float*)numa_alloc_onnode(M*N*sizeof(float), node);
//        ArowPtrData[node] = (int*)malloc((M+1)*sizeof(int));
//        AcolIndData[node] = (int*)malloc(Annz[node]*sizeof(int));
//        AvalData[node] = (float *)malloc(Annz[node]*sizeof(float));
//
//        BrowPtrData[node] = (int*)malloc((K/4+1)*sizeof(int));
//        BcolIndData[node] = (int*)malloc(Bnnz[node]*sizeof(int));
//        BvalData[node] = (float *)malloc(Bnnz[node]*sizeof(float));
//        CData[node] = (float*)malloc(M*N*sizeof(float));
    }
    for(int i=0; i<NUM_NODES; ++i){
        clearMatrix(CData[i], M, N);
    }

    DistributetoCSR(matrix, M, K, ArowPtrData, AcolIndData, AvalData);
    for (int i = 0; i < NUM_NODES; ++i) {
        toCSR(matrixB+i*(K/4*N), K/4, N, BrowPtrData[i], BcolIndData[i], BvalData[i]);
    }

    double start=0,end=0;
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
        threadArgs[i].ArowPtr = ArowPtrData[currentNode];
        threadArgs[i].AcolInd = AcolIndData[currentNode];
        threadArgs[i].Aval = AvalData[currentNode];
        threadArgs[i].BrowPtr = BrowPtrData[currentNode];
        threadArgs[i].BcolInd = BcolIndData[currentNode];
        threadArgs[i].Bval = BvalData[currentNode];
        threadArgs[i].C = CData[currentNode];
        threadArgs[i].M = M;
        threadArgs[i].K = K/NUM_NODES;
        threadArgs[i].N = N;
        threadArgs[i].threadPerNode = threads/NUM_NODES;
    }
    start = getHighResolutionTime();
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
    end = getHighResolutionTime();
    double sumUpTime = 0;
    double t1= getHighResolutionTime();
    sumUp(CData, M, N, threads);
    sumUpTime = getHighResolutionTime()-t1;
    double t = end-start;
    printf("compute time %lf s\n", t);
    printf("sumup time %lf s\n", sumUpTime);
    printf("total %lf\n", t+sumUpTime);

//    printf("%d %d %d %d\n", Annz[0], Annz[1], Annz[2], Annz[3]);

    // float* C = (float*)malloc(M*sizeof(float));
    // float* B = (float*)malloc(K*sizeof(float));
    // // for(int i=0; i<K; i++){
    // //     B[i]=2;
    // // }
    // for(int i=0; i<M; i++){
    //     C[i]=0;
    // }

//    toCSR(matrix, M, K, rowPtr1, colInd1, val1);
//    toCSR(matrixB, K, N, rowPtr2, colInd2, val2);
//    float* matrixC = (float*)malloc(M*N*sizeof(float));
//
//    double start,timeElapse;
//    start = getHighResolutionTime();
//    for(int i=0; i< ite; i++){
//        sparse_matrix_mul3(rowPtr1, colInd1, val1,
//                rowPtr2, colInd2, val2,
//                matrixC,M,K,N, threads);
//    }
//    timeElapse = getHighResolutionTime() - start;
//    cout << "M k N sparity: " << M << " " << K << " " << N << " " << ratio << endl;
//    cout << "timeTake: " << timeElapse/ite << endl;
    
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