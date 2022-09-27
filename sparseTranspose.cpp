#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <mutex>
#include <algorithm>
#include <string>
#include <string.h>
#include <vector>
struct Tuple{
    int row;
    int col;
    int val;
};
using namespace std;
double getHighResolutionTime(void) {
    struct timeval tod;
    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}
int prepareSparse2(int* matrix, int rows, int cols, int nnz, Tuple spM[]){
    int nozero = 0;
    for(int h=0;h<nnz;h++) {
        int i=rand()%rows;
        int j=rand()%cols;
        int num = (int)(rand()%100);
        matrix[i*cols + j] = num;
        Tuple tuple = {i,j,num};
        spM[nozero] = tuple;
        nozero++;

    }

    return nozero;
}
int prepareSparse(int* matrix, int rows, int cols, int nnz){
    int nozero = 0;
    for(int h=0;h<nnz;h++) {
        int i=rand()%rows;
        int j=rand()%cols;
        int num = (int)(rand()%100);
        matrix[i*cols + j] = num;
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


void fast_transpose(int* sm, int* res, mutex mtx[])
{
    //sm第一行   m,n,nnz_nums;
//    int total2[sm[0*3+0]], oriRowPtr[sm[0*3+0]+1];//原稀疏矩阵每行开头及结束位置
    int total[sm[0*3+1]], index[(sm[0*3+1]) + 1];
    for (int i = 0; i < sm[0*3+1]; i++){
        total[i] = 0;//原数组每列分别有多少个元素
        index[i] = 0;//每一行元素相对第一个元素的偏移

    }
//    for(int i=0; i< sm[0*3+0]; ++i){
//        total2[i] = 0;
//        oriRowPtr[i] = 0;
//    }

    for (int i = 1; i <= sm[0*3+2]; i++){
        total[sm[i*3+1]]++;
//        total2[sm[i*3+0]]++;
    }
    index[0] = 1;
//    oriRowPtr[0]=1;
    for (int i = 1; i <= sm[0*3+1]; i++){
        index[i] = index[i - 1] + total[i - 1];
    }
//    for(int i=1; i<=sm[0*3+0]; i++){
//        oriRowPtr[i] = oriRowPtr[i-1]+total2[i-1];
//    }
    res[0*3+0] = sm[0*3+1];
    res[0*3+1] = sm[0*3+0];
    res[0*3+2] = sm[0*3+2];

    double start = getHighResolutionTime();
    #pragma omp parallel for
    for (int i = 1; i <= sm[0*3+2]; i++)
    {
        mtx[sm[i*3+1]].lock();
        int loc = index[sm[i*3+1]];
        res[loc*3+0] = sm[i*3+1];
        res[loc*3+1] = sm[i*3+0];
        res[loc*3+2] = sm[i*3+2];
//        memcpy(res+loc*3, sm+i*3, sizeof(int)*3);
        index[sm[i*3+1]]++;
        mtx[sm[i*3+1]].unlock();
    }
    double elapse = getHighResolutionTime()-start;
    printf("valid time : %lf s\n", elapse);

}
void simpleTranspose(int* sm, int* res){
    res[0*3+0] = sm[0*3+1];
    res[0*3+1] = sm[0*3+0];
    res[0*3+2] = sm[0*3+2];
    #pragma omp parallel for
    for (int i = 1; i <= sm[0*3+2]; ++i) {
        res[i*3+0] = sm[i*3+1];
        res[i*3+1] = sm[i*3+0];
        res[i*3+2] = sm[i*3+2];
    }
}
bool cmp(const Tuple x,const Tuple y)
{
    if(x.row==y.row)
        return x.col<y.col;
    return x.row<y.row;
}
//void test(Tuple spM[], Tuple ans[], int M, int N, int nnz){
//    #pragma omp parallel for
//    for (int i = 0; i < nnz; ++i) {
//        printf("%d\n", omp_get_num_threads());
//        ans[i].row = spM[i].col;
//        ans[i].col = spM[i].row;
//        ans[i].val = spM[i].val;
//    }
//    sort(ans, ans+nnz, cmp);
//}


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

void clear(int* A, int m, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            A[i*n+j]=0;
        }
    }
    return;
}

int main(int argc, char *argv[]){

    int M = strtol(argv[1], NULL, 10);
    int N = strtol(argv[2], NULL, 10);
    int threads = strtol(argv[3], NULL, 10);
//    int M=100,N=100,threads=4;
    omp_set_num_threads(threads);
    // int M=10;
    // int N=15;
    float ratio = 0.01;
    mutex mtx[N];
    int* A = (int*)malloc(M*N*sizeof(int));
    clear(A, M, N);
    int nozero = (int)(ratio*M*N);
    printf("nnz %d", nozero);
    int cap = (nozero+1)*3;
    int* sm_A = (int*)malloc(cap*sizeof(int));
    int* res = (int*)malloc(cap*sizeof(int));
    cout << "t:" << nozero <<endl;
    srand((unsigned)time(0));
//    struct Tuple ans[nozero];
//    struct Tuple spM[nozero];
//    struct Tuple* ans = (struct Tuple*)malloc(sizeof(struct Tuple)*nozero);
//    struct Tuple* spM = (struct Tuple*)malloc(sizeof(struct Tuple)*nozero);

//    prepareSparse2(A, M, N, nozero, spM);
     prepareSparse(A, M, N, nozero);

//    display(A, M, N);
    convert(A, sm_A, M, N);
    // convert(B, sm_B, M, N);

    double time1=getHighResolutionTime();
    fast_transpose(sm_A, res, mtx);
//    simpleTranspose(sm_A, res);
//    test(spM, ans, M, N, nozero);
    double timeTake = getHighResolutionTime()-time1;

    cout << "TimeTake: " << timeTake << " s" << endl;
//     displaySparse(res);
//     displaySparse(sm_A);
    return 0;

}