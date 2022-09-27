//g++ -g -O3 rowBaseSpmm.cpp -o rowBaseSpmm -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas -fopenmp
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>
#include <arm_neon.h>
#include <fstream>
#include <vector>
#include <string.h>
#include <string>
using namespace std;

double getHighResolutionTime(void);
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
      matrix[i*cols + j] = (float)(rand()%20000-10000)/1000;
    }
    
    // toCSR(matrix, rows, cols, rowPtr, colInd, val);
    
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

int main(int argc, char *argv[]){
    //suitesparse dataset
    string path = "datasets/mouse_gene.mtx";
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
        
        nums = split(str, " ");
        int i=atoi(nums[0].c_str());
        int j=atoi(nums[1].c_str());
        float val = atof(nums[2].c_str());
        // cout << str << endl;
        
        matrix[(i-1)*cols + (j-1)] = val;
    }
    int threads = strtol(argv[1], NULL, 10);
    int ite = 30;
    int M = rows;
    int K = cols;
    cout << "1";
    float* C = (float*)malloc(M*sizeof(float));
    float* B = (float*)malloc(K*sizeof(float));
    for(int i=0; i<K; i++){
        B[i]=5;
    }
    for(int i=0; i<M; i++){
        C[i]=0;
    }
    int* rowPtr = (int *)malloc(4*nnz*sizeof(int));
    int* colInd = (int *)malloc(4*nnz*sizeof(int));
    float* val =  (float *)malloc(4*nnz*sizeof(float));
    cout << "2";

    toVectorAwareCSR(matrix, M, K, rowPtr, colInd, val);
    sparse_dense_vector(B, C, rowPtr, colInd, val, M, threads);
    cout <<"3";
    double start,end,elapse;
    start = getHighResolutionTime();
    for(int i=0;i <ite; i++){
        sparse_dense_vector(B, C, rowPtr, colInd, val, M, threads);
    }
    
    end = getHighResolutionTime();
    elapse = end - start;
    cout << "m k: " << M << " " << K << " "<<endl;;
    cout << "time take: " << elapse/ite << endl;
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
    // // show Sparse
    // cout << "nozero:" << nozero << endl;
    // for(int i=0;i<4*nozero;i++) cout << rowPtr[i] << " ";
    // cout << endl;
    // for(int i=0;i<4*nozero;i++) cout << colInd[i] << " ";
    // cout << endl;
    // for(int i=0;i<4*nozero;i++) cout << val[i] << " ";
    // cout << endl;

    // cout << "B-vector:" << endl;
    // for(int i=0;i<K;i++) cout << B[i] << " ";
    // cout << endl;
    // cout << "C-vector:" << endl;
    // for(int i=0;i<M;i++) cout << C[i] << " ";
    // cout << endl;
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
void sparse_dense_vector(float *B, float *C, int rowPtr[], int colInd[], float val[], int M,int threads){

    #pragma omp parallel for num_threads(threads)
    for(int m=0; m<M; m++){
        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        
        float32x4_t sum_vec=vdupq_n_f32(0),left_vec,right_vec;
        for(int idx=row_start; idx<row_end; idx+=4){
            int k=colInd[idx];

            left_vec = vld1q_f32(val+idx);
            right_vec=vld1q_f32(B+k);
            sum_vec=vmlaq_f32(sum_vec,left_vec,right_vec);
            // int k=colInd[idx];
            // float* a = &val[idx];
            // float* b = &B[k];
            // float* c = &C[m];
            // vector_vector_neon(a, b, c);
        }  
        float32x2_t r=vadd_f32(vget_high_f32(sum_vec),vget_low_f32(sum_vec));
        C[m]+=vget_lane_f32(vpadd_f32(r,r),0);

    }
}
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}