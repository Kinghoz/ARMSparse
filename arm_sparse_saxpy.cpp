#include <stdio.h>
#include <arm_neon.h>
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <cblas.h>
double getHighResolutionTime(void) {
    struct timeval tod;
    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}
void num_vector_neon(float32_t *A, float32_t *B, float32_t *C, int len){
    // num-vector mul c = c+a*b (b is a scalar)
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
void scalaxvector(float *A,const float32_t b, const size_t n){
    //a = a*b 
    float32x4_t c;
    float32x4_t a;
    size_t i;
    for(i = 0; i+4<=n; i+=4){
        // c = vmovq_n_f32(0);
        a = vld1q_f32(A+i);
        // c = vfmaq_n_f32(c, a, b);
        // vst1q_f32(A+i, c);
        c = vmulq_n_f32(a, b);
        vst1q_f32(A+i, c);
    }
    for(; i<n; ++i){
        A[i] = A[i]*b;
    }
    return;
}
void arm_sparse_saxpy(const int nz, const float a, float *x, const int *indx, float *y){
    //乘加和 y = y + x*a
    // num_vector_neon(x, &a, x, nz);
    scalaxvector(x, a, nz);

    for(int i=0; i<nz; i++){
        y[indx[i]] += x[i];
    }

    return;
}
void arm_sparse_sdoti(const int nz, const float *x,const int *indx, const float *y, float *dot){
    //todo向量加和,內积
    size_t i;
    float32x4_t acc;
    float32x4_t t;
    for(i = 0; i<nz; i++){
        *dot += x[i]*y[indx[i]];
    }
    return;
}
void arm_sparse_sgthr(const int nz, const float *y, float *x, const int *indx){
    //将full-storage格式的稀疏向量y中指定的元素加载到compressed格式的向量中。即x[i]=y[indx[i]]，i=0,1,…nz-1。
    for(size_t i=0; i<nz; i++){
        x[i] = y[indx[i]];
    }
    return;
}
void arm_sparse_ssctr(const int nz, const float *x, const int *indx, float *y){
    //将compressed格式的向量写入full-storage格式的稀疏向量y指定位置。即y[indx[i]]=x[i]，i=0,1,…nz-1。
    for(size_t i =0; i<nz; i++){
        y[indx[i]]=x[i];
    }
}
int main(int argc, char* argv[]){
    int m = strtol(argv[1], NULL, 10);
    float ratio = 0.1;
    int ite = 1;
    int nnz = ratio*m;
    float *res = (float *)malloc(m*sizeof(float ));
    float *x = (float *)malloc(nnz*sizeof(float ));
    int *indx = (int*)malloc(nnz*sizeof(int));
    for(int i=0; i<nnz; ++i){
        x[i] = (float)(rand()%100);
        indx[i] = (int)(rand()%m);
    }


    double start,end,elapse;
    start = getHighResolutionTime();
    for(int i=0; i<ite; ++i){
        arm_sparse_saxpy(nnz, 0.3, x, indx, res);
    }
    double saxpyTime = getHighResolutionTime()-start;
    float dotRes=0.0;
    float *y = (float *)malloc(m*sizeof(float ));
    for(int i=0; i<m; ++i) y[i] = 3.0;
    start = getHighResolutionTime();
    for(int i=0; i<ite; ++i){
        arm_sparse_sdoti(nnz, x, indx, y, &dotRes);
    }
    double dotiTime = getHighResolutionTime()-start;
    start = getHighResolutionTime();
    float * denseX = (float *)malloc(m*sizeof(float ));
    for (int i = 0; i < m; ++i) {
        denseX[i] = 0.0;
    }

    for(int i=0; i<ite; ++i){
        cblas_saxpy(m, 0.3, denseX, 1, res, 1);
    }
    double cblasSaxpyTime = getHighResolutionTime()-start;
    printf("m: %d, ratio: %d\n",m, ratio);
    printf("saxpy timeTake: %lf\n", saxpyTime/ite);
    printf("doti timeTake: %lf\n", dotiTime/ite);
    printf("cblassaxpy timeTake: %lf\n", cblasSaxpyTime/ite);

}