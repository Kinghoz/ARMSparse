#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <ctime>
#include <iostream>
#include <string.h>
#include <omp.h>
using namespace std;
void toCSR(float* matrix, int rows,int cols, int* rowPtr, int* colInd, float* val);
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}
int prepareSparse(int* matrix, int rows, int cols, float ratio){
    int nozero = 0;

    for(int h=0;h<ratio*rows*cols;h++) {
        int i=rand()%rows;
        int j=rand()%cols;
        matrix[i*cols + j] = (int)(rand()%10);
        nozero++;

    }
    return nozero;
}

void display(int *C, int m, int n)
{
    printf("\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("\t%d", *C++);
        printf("\n");
    }
    printf("\n");
}
void displaySparse(int *A){
	printf("displaySparse:\n");
	for(int i=0; i<A[2]+1; i++){
		printf("%d %d %d\n", A[i*3+0], A[i*3+1], A[i*3+2]);
	}
	return;
}

int T[100][3];
int T1[100][3];

void convert(int *A, int *sm_A, int m, int n)
{
    sm_A[0*3+0] = m;
    sm_A[0*3+1] = n;

    int k = 1;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            if (A[i*n+j] != 0)
            {
                sm_A[k*3+0] = i;
                sm_A[k*3+1] = j;
                sm_A[k*3+2] = A[i*n+j];
                ++k;

            }

        }
    sm_A[0*3+2] = k - 1;
	// displaySparse(sm_A, k);
}

void _convert(int *A, int m, int n)
{
    T1[0][0] = m;
    T1[0][1] = n;
    int k = 1;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            if (*(A) != 0)
            {
                T1[k][0] = i;
                T1[k][1] = j;
                T1[k++][2] = *A;
            }
            *(A++);
        }
    T1[0][2] = k - 1;

    display(T1[0], k, 3);
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

void add(int *sm_A,int *sm_B, int *add, int m, int n)
{

    int c1 = 1, c2 = 1, cm = 0;
    add[0*3+0] = m;
    add[0*3+1] = n;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int flag = 0;
            if (sm_A[c1*3+0] == i && sm_A[c1*3+1] == j)
            {
                cm++;
                add[cm*3+0] = i;
                add[cm*3+1] = j;
                add[cm*3+2] = sm_A[c1*3+2];
                c1++;
                flag = 1;
            }
            if (sm_B[c2*3+0] == i && sm_B[c2*3+1] == j)
            {
                if (!flag)
                {
                    cm++;
                    add[cm*3+2] = 0;
                }
                add[cm*3+0] = i;
                add[cm*3+1] = j;
                add[cm*3+2] += sm_B[c2*3+2];
                c2++;
            }
        }
    }
    add[0*3+2] = cm;
    // display(add, cm + 1, 3);
}
void addition(int *sm_A,int *sm_B, int *res, int m, int n){
    int endA = sm_A[2]+1, endB=sm_B[2]+1;
    int rowPtrA = 1, rowPtrB = 1;
    int cnt = 1;
    res[0]=m,res[1]=n;
    while(rowPtrA < endA || rowPtrB < endB){
        int t = cnt*3, t1 = rowPtrA*3, t2=rowPtrB*3;
        int coor1 = rowPtrA<endA ? sm_A[t1]*n+sm_A[t1+1]:m*n;
        int coor2 = rowPtrB<endB ? sm_B[t2]*n+sm_B[t2+1]:m*n;
		
        if(coor1 == coor2){
            res[t]=sm_A[t1];
            res[++t]=sm_A[++t1];
            res[++t]=sm_A[++t1]+sm_B[t2+2];
            ++rowPtrA;
            ++rowPtrB;
        }
        else if(coor1 < coor2){
            res[t]=sm_A[t1];
            res[++t]=sm_A[++t1];
            res[++t]=sm_A[++t1];
            ++rowPtrA;
        }
        else{
            res[t]=sm_B[t2];
            res[++t]=sm_B[++t2];
            res[++t]=sm_B[++t2];
            ++rowPtrB;
        }
        ++cnt;
    }
    res[2]=cnt-1;
}

void csradd(int* rowPtr, int* colInd, float* val,float* matrix,int M,int N){
    #pragma omp parallel for
    for(int m=0; m<M; m++){

        int row_start = rowPtr[m];
        int row_end = rowPtr[m+1];
        for(int idx=row_start; idx<row_end; idx++){
            int k=colInd[idx];
            matrix[m*N+k] += val[idx];
        }

    }

}
void clear(int * A, int m, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            A[i*n+j]=0;
        }
    }
    return;
}

void test(int *rowPtr1,int *colInd1, float *val1, int *rowPtr2,int *colInd2, float *val2, int M, int N, float* res){

    double start = getHighResolutionTime();
    csradd(rowPtr1, colInd1, val1, res, M, N);
    csradd(rowPtr2, colInd2, val2, res, M, N);
    double timeTake = getHighResolutionTime()-start;
    printf("TimeTake %lf s\n", timeTake);
}


int main(int argc, char *argv[]){
  int M = strtol(argv[1], NULL, 10);
  int N = strtol(argv[2], NULL, 10);
  int threads = strtol(argv[3], NULL, 10);
	//  int M=3;
    //  int N=3;
    omp_set_num_threads(threads);
    float ratio = 0.1;
//    float* A = (float*)malloc(M*N*sizeof(float));
//    float * B = (float *)malloc(M*N*sizeof(float ));
//    float* C = (float *)malloc(M*N*sizeof(float ));
    int* A = (int*)malloc(M*N*sizeof(int));
    int * B = (int *)malloc(M*N*sizeof(int ));
//    int* C = (int *)malloc(M*N*sizeof(int ));
    clear(A, M, N);
    clear(B, M, N);
//    clear(C, M, N);
    int nozero = (int)(ratio*M*N);
    int cap = (nozero+1)*3;
    int* sm_A = (int*)malloc(cap*sizeof(int));
    int* sm_B = (int*)malloc(cap*sizeof(int));
    int* res = (int*)malloc(2*cap*sizeof(int));
//    int* rowPtr1 = (int *)malloc((M+1)*sizeof(int));
//    int* colInd1 = (int*)malloc(nozero*sizeof(int));
//    float* val1 = (float *)malloc(nozero*sizeof(float ));
//    int* rowPtr2 = (int *)malloc((M+1)*sizeof(int));
//    int* colInd2 = (int*)malloc(nozero*sizeof(int));
//    float* val2 = (float *)malloc(nozero*sizeof(float ));
    cout << "t: " << nozero <<endl;
    srand((unsigned)time(0));
    prepareSparse(A, M, N, ratio);
    prepareSparse(B, M, N, ratio);
//    toCSR(A, M, N, rowPtr1, colInd1, val1);
//    toCSR(B, M, N, rowPtr2, colInd2, val2);
//    for(int i=0; i<M; i++){
//        for(int j=0; j<N; j++){
//            A[i*N+j]=3;
//            B[i*N+j]=3;
//        }
//    }
//      display(A, M, N);
//      display(B, M, N);
    convert(A, sm_A, M, N);
    convert(B, sm_B, M, N);
//      displaySparse(sm_A);
//      displaySparse(sm_B);
    double time1=getHighResolutionTime();
//    test(rowPtr1, colInd1, val1,rowPtr1, colInd1, val1, M, N, C);
    addition(sm_A, sm_B, res, M, N);
    double timeTake = getHighResolutionTime()-time1;

//    int* rowPtr3 = (int *)malloc((M+1)*sizeof(int));
//    int* colInd3 = (int*)malloc(nozero*sizeof(int));
//    float* val3 = (float *)malloc(nozero*sizeof(float ));
    double time2 = getHighResolutionTime();
//    toCSR(C, M, N, rowPtr3, colInd3, val3);
    double toCSRTime = getHighResolutionTime()-time2;
    cout << "TimeTake: " << timeTake << " s" << endl;
    printf("toCSRTime %lf s\n", toCSRTime);
//      displaySparse(res);
    return 0;

}