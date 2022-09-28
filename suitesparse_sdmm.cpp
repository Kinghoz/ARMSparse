//g++ -g -O3 suitesparse.cpp -o suitesparse -I $PATHOFOPENBLAS/include -L $PATHOFOPENBLAS/lib -lopenblas -fopenmp
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
void toDataReuseFormat(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val, bool* flag);
void DataReuseSpmm(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads, bool* flag){
    omp_set_num_threads(threads);

    int n = N;
    #pragma omp parallel for schedule(static,2)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];

        for(int idx=row_start; idx<row_end; idx++){
            int k=colInd[idx];
            bool isnextrow=flag[idx];

            cblas_saxpy(n, val[idx], B+(k*N), 1, C+((m+isnextrow)*N), 1);

        }
      
    }

}
void rowBaseSpmm(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads){

    int n = N;
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        
        for(int idx=row_start; idx<row_end; idx++){
            int k=colInd[idx];

            cblas_saxpy(n, val[idx], B+(k*N), 1, C+(m*N), 1);
            // num_vector_neon(B+(k*N), val+idx, C+(m*N), n);
        }      
    }

}

int main(int argc, char *argv[]){
    //suitesparse dataset
    string path = "datasets/wang3.mtx";
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
    printf("%d %d %d\n", rows, cols, nnz);

    while(getline(fp, str)){
        // cout << str << endl;
        nums = split(str, " ");
        int i=atoi(nums[0].c_str());
        int j=atoi(nums[1].c_str());
        float val = atof(nums[2].c_str());

        
        matrix[(i-1)*cols + (j-1)] = val;
    }
    int threads = strtol(argv[1], NULL, 10);
    int ite = 30;
    int M = rows;
    int K = cols;
    int N = 1024;
    

    int* rowPtr = (int *)malloc(nnz*sizeof(int));
    int* colInd = (int *)malloc(nnz*sizeof(int));
    float* val =  (float *)malloc(nnz*sizeof(float));
    

    
    float *B = (float *)malloc(K*N*sizeof(float));
    float *C = (float *)malloc(M*N*sizeof(float));
    float *C_compare = (float *)malloc(M*N*sizeof(float));
    float *C_compare2 = (float *)malloc(M*N*sizeof(float));
    float *C_compare3 = (float *)malloc(M*N*sizeof(float));
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            C[i*N+j]=0.f;
            C_compare[i*N+j]=0.f;
            C_compare2[i*N+j]=0.f;
            C_compare3[i*N+j]=0.f;
        }
    }
    float num=1;
    for(int i=0;i<K;i++){
        for(int j=0;j<N;j++){
            // B[i*N+j]=(float)(rand()%20000-10000)/1000;
            B[i*N+j]=(float)(rand()%10);
        }        
    }
    

    // DataReuseFormat
    int* rowPtr2 = (int *)malloc(nnz*sizeof(int));
    int* colInd2 = (int *)malloc(nnz*sizeof(int));
    float* val2 =  (float *)malloc(nnz*sizeof(float));
    bool* flag = (bool *)malloc(nnz*sizeof(bool));

    toCSR(matrix, M, K, rowPtr, colInd, val);
    toDataReuseFormat(matrix, M, K, rowPtr2, colInd2, val2, flag);
    

    rowBaseSpmm(rowPtr, colInd, val, B, C_compare, M, K, N, threads);
    double timeStart = getHighResolutionTime();
    for(int i=0; i<ite; i++){
        rowBaseSpmm(rowPtr, colInd, val, B, C_compare2, M, K, N, threads);
    }
    double rowBaseSpmmTime = getHighResolutionTime()-timeStart;


    cout <<"m k n ratio:" << M<<" "<< K <<" "<< N  << "  nnz:" << nnz  <<endl;
    cout << "rowBaseSpmmTime " << rowBaseSpmmTime/ite << endl;

    return 0;
    
}
void toDataReuseFormat(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val, bool* flag){
    int size = 0;
    int max_tile = 2;
    for(int row=0;row < rows;row++){
        rowPtr[row] = size;
        for(int col=0;col < cols; col++){
            float num=matrix[row*cols + col];
            
            if(num != 0){
                val[size]=num;
                colInd[size] = col;
                flag[size] = false;
                size++;
                for(int cur_row=row+1; (cur_row)%max_tile!=0; cur_row++){
                    
                    if(cur_row < rows && matrix[cur_row*cols+col] != 0){
                        val[size] = matrix[cur_row*cols+col];
                        colInd[size] = col;
                        flag[size] = true;
                        size++;
                        matrix[cur_row*cols+col] = 0;
                    }else{
                        break;
                    }
                }
            }
        }
    }
    rowPtr[rows] = size;
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

    int n = N;
    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        // float* buff = (float*)malloc(n*sizeof(float));
        for(int idx=row_start; idx<row_end; idx++){
            int k=colInd[idx];

            cblas_saxpy(n, val[idx], B+(k*N), 1, C+(m*N), 1);
            // cblas_saxpy(n, val[idx], B+(k*N), 1, buff, 1);
        }
        // cblas_scopy(n, buff, 1, C+(m*N), 1);
    }

}
vector<string> split(const string& str, const string& delim) {  
    vector<string> res;  
    if("" == str) return res;  

    char * strs = new char[str.length() + 1] ; 
    strcpy(strs, str.c_str());   

    char * d = new char[delim.length() + 1];  
    strcpy(d, delim.c_str());  

    char *p = strtok(strs, d);  
    while(p) {  
        string s = p; 
        res.push_back(s); 
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
