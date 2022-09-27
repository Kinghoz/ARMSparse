//g++ suitesparse.cpp -o suitesparse -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -fopenmp
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>
#include <string>
#include <fstream>
#include <vector>
#include <string.h>
#include <arm_neon.h>

using namespace std;
void num_vector_neon(float32_t *A, float32_t *B, float32_t *C, int len);
vector<string> split(const string& str, const string& delim);
void origin(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads);
void rowBaseSpmm_blas(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads);
void prepareSparse(float* matrix, int rows, int cols,int* rowPtr, int* colInd, float* val, float ratio);
void toCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val);
double getHighResolutionTime(void);
void mul(int m, int k, int n, float* A, float* B, float* C);
bool compare(int m,int n, float* C1, float* C2);
void sparse_matrix_mul(int rowPtr1[], int colInd1[], float val1[],
                int rowPtr2[], int colInd2[], float val2[], 
                int rowPtr3[], int colInd3[], float val3[],
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
    toCSR(matrixC, M, N, rowPtr3, colInd3, val3);
    
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

int main(int argc, char *argv[]){
    //suitesparse dataset
    string path = "datasets/big.mtx";
    ifstream fp(path);
    std::ifstream fin(path);
    string str;
    vector<string> nums;
    while(getline(fp, str)){
        if(str[0]!='%') break;
    }
    nums = split(str, " ");
    int rows,cols,nnz;
    rows = atoi(nums[0].c_str());
    cols = atoi(nums[1].c_str());
    nnz = atoi(nums[2].c_str());
    float* matrix = (float*)malloc(rows*cols*sizeof(float));
    float* matrixb = (float*)malloc(rows*cols*sizeof(float));
    for(int i=0; i<rows*cols; ++i){matrix[i] = 0;}
    for(int i=0; i<rows*cols; ++i){matrixb[i] = 0;}
    printf("%d %d %d\n", rows, cols, nnz);

    while(getline(fp, str)){
        
        nums = split(str, " ");
        int i=atoi(nums[0].c_str());
        int j=atoi(nums[1].c_str());
        float val = atof(nums[2].c_str());
        // cout << str << endl;
        
        matrix[(i-1)*cols + (j-1)] = val;
        matrixb[(i-1)*cols + (j-1)] = val;
    }
    int threads = strtol(argv[1], NULL, 10);
    int ite = 10;
    int M = rows;
    int K = cols;
    int N = rows;
    

    int* rowPtr1 = (int *)malloc((M+1)*sizeof(int));
    int* colInd1 = (int *)malloc(nnz*sizeof(int));
    float* val1 =  (float *)malloc(nnz*sizeof(float));
    int* rowPtr2 = (int *)malloc((K+1)*sizeof(int));
    int* colInd2 = (int *)malloc(nnz*sizeof(int));
    float* val2 =  (float *)malloc(nnz*sizeof(float));
    int* rowPtr3 = (int *)malloc((M+1)*sizeof(int));
    int* colInd3 = (int *)malloc(nnz*sizeof(int));
    float* val3 =  (float *)malloc(nnz*sizeof(float));

    float *C = (float *)malloc(M*N*sizeof(float));


    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            C[i*N+j]=0.f;
        }
    }
    float num=1;
    toCSR(matrix, M, K, rowPtr1, colInd1, val1);
    toCSR(matrixb, K, N, rowPtr2, colInd2, val2);
    // sparse_matrix_mul3(rowPtr1, colInd1, val1,rowPtr2, colInd2, val2, 
    //             C, M, K, N, threads);
    sparse_matrix_mul(rowPtr1, colInd1, val1,rowPtr2, colInd2, val2,rowPtr3, colInd3, val3, 
                C, M, K, N, threads);

    double timeStart = getHighResolutionTime();
    for(int i=0; i<ite; i++){
        sparse_matrix_mul3(rowPtr1, colInd1, val1,
                rowPtr2, colInd2, val2, 
                C, M, K, N, threads);
        // sparse_matrix_mul(rowPtr1, colInd1, val1,rowPtr2, colInd2, val2,rowPtr3, colInd3, val3, 
        //         C, M, K, N, threads);
    }
    double SpMMTime = getHighResolutionTime()-timeStart;
    timeStart = getHighResolutionTime();
    toCSR(C, M, N, rowPtr3, colInd3, val3);
    double tocsrTime = getHighResolutionTime()-timeStart;
    cout << "tocsrTime: " << tocsrTime << endl;
    cout <<"m k n ratio:" << M<<" "<< K <<" "<< N  << "  nnz:" << nnz  <<endl;
    // cout << "origin_timeTake:" << origin_timeTake/ite << endl;
    cout << "SpMMTime " << SpMMTime/ite << endl;
    // show result C


    return 0;
    


}
void prepareSparse(float* matrix, int rows, int cols,int* rowPtr, int* colInd, float* val, float ratio){

    // srand((unsigned)time(NULL));
    srand(4);
    int nnz = ratio*rows*cols;
    for(int h=0; h<nnz; h++) {
      int i=rand()%rows;
      int j=rand()%cols;
      if(matrix[i*cols + j] != 0){
          h--;
          continue;
      }
    //   matrix[i*cols + j] = (float)(rand()%20000-10000)/1000;
        matrix[i*cols + j] = (float)(rand()%10);
    }
    
    toCSR(matrix, rows, cols, rowPtr, colInd, val);
    
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
    float epsilon = 1;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if( abs(C1[i*n+j] - C2[i*n+j]) > epsilon ){

                
                return false;
            } 
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
                C[i*N+j] += val[ptr] * B[t*N+j];
            }
            // C[i*N+j] = res;
        }
        
    }

}
void rowBaseSpmm_blas(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){
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
        // float* buff = (float*)malloc(n*sizeof(float));
        for(int idx=row_start; idx<row_end; idx++){
            int k=colInd[idx];
            // for(int n=0; n<N; n++){
            //     C[m*N+n] += val[idx]*B[k*N+n]; 
            // }
            cblas_saxpy(n, val[idx], B+(k*N), 1, C+(m*N), 1);
            // cblas_saxpy(n, val[idx], B+(k*N), 1, buff, 1);
        }
        // cblas_scopy(n, buff, 1, C+(m*N), 1);
    }

}
vector<string> split(const string& str, const string& delim) {  
    vector<string> res;  
    if("" == str) return res;  
    //先将要切割的字符串从string类型转换为char*类型  
    char * strs = new char[str.length() + 1] ; //不要忘了  
    strcpy(strs, str.c_str());   

    char * d = new char[delim.length() + 1];  
    strcpy(d, delim.c_str());  

    char *p = strtok(strs, d);  
    while(p) {  
        string s = p; //分割得到的字符串转换为string类型  
        res.push_back(s); //存入结果数组  
        p = strtok(NULL, d);  
    }   
    return res;  
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
