//g++ -I /home/nscc-gz/zhengj/work/eigen-3.3.9 eigenSparseVector.cpp -o eigenSparseVector -O3 -fopenmp
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <vector>
#include <iostream>
#include <sys/time.h>
using namespace std;
using namespace Eigen;

typedef SparseMatrix<float, RowMajor> spMatFloat;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> deMatRowFloat;
typedef Matrix<float,Dynamic,1> DenseVector;


double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}

int main(int argc, char** argv)
{
    int m = strtol(argv[1], NULL, 10);
    SparseVector<float ,RowMajor> vec(m);
    SparseVector<float ,RowMajor> res(m);

    float ratio = 0.1;
    int ite = 1;
    int nnz = ratio*m;
    for(int i=0; i<nnz; ++i){
        int idx = (int)(rand()%m);
        float num = (float)(rand()%10);
        vec.coeffRef(idx) = num;
    }


    double start,end,elapse;
    start = getHighResolutionTime();

    res += vec * 0.3;

    end = getHighResolutionTime();
    double saxpyTime = end-start;
    SparseVector<float ,RowMajor> a(m);
    SparseVector<float ,ColMajor> b(m);
    SparseVector<float ,RowMajor> c(m);
//     a*b.transpose();
    start = getHighResolutionTime();
    c = a*b;
    double dotiTime = getHighResolutionTime()-start;

    
    cout << "m  ratio: " << m << " " << ratio << endl;
    cout << "saxpy time take: " << saxpyTime << endl;
    cout << "doti time take: " << dotiTime << endl;

 
  
 
    return 0;
}