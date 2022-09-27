//g++ data_reuse_format.cpp -o data_reuse_format -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -fopenmp
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

using namespace std;
vector<string> split(const string& str, const string& delim);
void origin(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads);
void rowBaseSpmm(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads);
void prepareSparse(float* matrix, int rows, int cols,int* rowPtr, int* colInd, float* val, float ratio);
void toCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val);
double getHighResolutionTime(void);
void mul(int m, int k, int n, float* A, float* B, float* C);
bool compare(int m,int n, float* C1, float* C2);
void toDataReuseFormat(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val, bool* flag);
void DataReuseSpmm(int rowPtr[], int colInd[], float val[], float B[], float C[], int M, int K, int N, int threads, bool* flag){
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
            bool isnextrow=flag[idx];
            // for(int n=0; n<N; n++){
            //     C[m*N+n] += val[idx]*B[k*N+n]; 
            // }
            // cblas_saxpy(n, val[idx], B+(k*N), 1, C+(m*N), 1);
            cblas_saxpy(n, val[idx], B+(k*N), 1, C+((m+isnextrow)*N), 1);
            // cblas_saxpy(n, val[idx], B+(k*N), 1, buff, 1);
        }
        // cblas_scopy(n, buff, 1, C+(m*N), 1);
    }

}


int main(int argc, char *argv[]){
    
    string path = "datasets/cage4.mtx";
    ifstream fp(path);
    std::ifstream fin(path);
    string str;
    vector<string> nums;
    cout << -1;
    while(getline(fp, str)){
        if(str[0]!='%') break;
        
    }
    cout << 0.5;
    nums = split(str, " ");
    int rows,cols,nnz;
    rows = atoi(nums[0].c_str());
    cols = atoi(nums[1].c_str());
    nnz = atoi(nums[2].c_str());
    float* matrix = (float*)malloc(rows*cols*sizeof(float));
    printf("%d %d %d\n", rows, cols, nnz);
    int cnt=0;
    cout << 0;
    for(int idx=0; idx<nnz; idx++){
        cout << cnt;
        getline(fp, str);
        nums = split(str, " ");
        int i=atoi(nums[0].c_str());
        int j=atoi(nums[1].c_str());
        float val = atof(nums[2].c_str());
        cnt++;
        matrix[(i-1)*cols + (j-1)] = val;
        
        
    }
    int M = rows;
    int K = cols;
    int N = 4;

    // int M = strtol(argv[1], NULL, 10);
    // int K = strtol(argv[2], NULL, 10);
    // int N = strtol(argv[3], NULL, 10);
    int threads = strtol(argv[1], NULL, 10);
    int ite = 1;
    
    // int M=1000,K=1000,N=1000;
    float ratio = 0.5;
    // int nozero = (int)(ratio*M*K);
    // float* matrix = (float*)malloc(M*K*sizeof(float));
    int* rowPtr = (int *)malloc(nnz*sizeof(int));
    int* colInd = (int *)malloc(nnz*sizeof(int));
    float* val =  (float *)malloc(nnz*sizeof(float));
    cout << "1";
    prepareSparse(matrix, M, K, rowPtr, colInd, val, ratio);
    
    float *B = (float *)malloc(K*N*sizeof(float));
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
            // B[i*N+j]=(float)(rand()%20000-10000)/1000;
            B[i*N+j]=(float)(rand()%10);
        }        
    }
    cout << 2;
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
    // DataReuseFormat
    int* rowPtr2 = (int *)malloc(nnz*sizeof(int));
    int* colInd2 = (int *)malloc(nnz*sizeof(int));
    float* val2 =  (float *)malloc(nnz*sizeof(float));
    bool* flag = (bool *)malloc(nnz*sizeof(bool));
    // cout << "Matrix" << endl;
    // for(int i=0; i<M; i++){
    //     for(int j=0; j<K; j++){
    //         // cout << matrix[i*K+j] << " ";
    //         printf("%.1f ", matrix[i*K+j]);
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // cout << "B[]" << endl;
    // for(int i=0; i<K; i++){
    //     for(int j=0; j<N; j++){
    //         // cout << matrix[i*K+j] << " ";
    //         printf("%.1f ", B[i*N+j]);
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    toCSR(matrix, M, K, rowPtr, colInd, val);
    toDataReuseFormat(matrix, M, K, rowPtr2, colInd2, val2, flag);
    
    // show Sparse
    // cout << "show sparse" << endl;
    // for(int i=0;i<nozero;i++) cout << rowPtr2[i] << " ";
    // cout << endl;
    // for(int i=0;i<nozero;i++) cout << colInd2[i] << " ";
    // cout << endl;
    // for(int i=0;i<nozero;i++) printf("%.1f ", val2[i]);
    // cout << endl;
    // for(int i=0;i<nozero;i++) cout << flag[i] << " ";
    // cout << endl;
    // // origin(rowPtr, colInd, val, B, C_compare, M, K, N, threads);
    double t1 = getHighResolutionTime();
    for(int i=0; i<ite; i++){
        DataReuseSpmm(rowPtr2, colInd2, val2, B, C, M, K, N, threads, flag);
    }
    double t2 = getHighResolutionTime();
    double take = t2-t1;
    double timeStart = getHighResolutionTime();
    for(int i=0; i<ite; i++){
        rowBaseSpmm(rowPtr, colInd, val, B, C_compare2, M, K, N, threads);
    }
    double rowBaseSpmmTime = getHighResolutionTime()-timeStart;

    double t3 = getHighResolutionTime();
    for(int i=0; i<ite; i++){
        origin(rowPtr, colInd, val, B, C_compare, M, K, N, threads);
    }
    
    double t4 = getHighResolutionTime();
    double origin_timeTake = t4-t3;
    bool same = compare(M,N,C,C_compare);
    if(same){
        cout << "correct result!" << endl;
    }else{
        cout << "false result!" << endl;
    }
    cout <<"m k n ratio:" << M<<" "<< K <<" "<< N  << "  nnz:" << nnz  << "  Time take:" << take/ite << endl;
    cout << "origin_timeTake:" << origin_timeTake/ite << endl;
    cout << "rowBaseSpmmTime " << rowBaseSpmmTime/ite << endl;
    // // show result C
    // cout <<endl << "C[]" <<endl;
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
                // if(row < rows && matrix[(row+1)*cols+col]!=0){
                //     val[size] =  matrix[(row+1)*cols+col];
                //     colInd[size] = col;
                //     size++;
                //     matrix[(row+1)*cols+col] = 0;
                // }
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
    float epsilon = 0.001;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if( abs(C1[i*n+j] - C2[i*n+j]) > epsilon ) return false;
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