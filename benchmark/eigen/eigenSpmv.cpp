//g++ -I ~/zhengj/work/eigen-3.3.9 eigenSpmv.cpp -o eigenSpmv -O3 -fopenmp
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
    int n = strtol(argv[2], NULL, 10);
    int threads = strtol(argv[3], NULL, 10);
    Eigen::setNbThreads(threads);
    float ratio = 0.6;
    int ite = 30;
    DenseVector b(n), c(m);
    // Eigen::VectorXd b(n);   
    // Eigen::VectorXd c(m);      
    deMatRowFloat d_m=deMatRowFloat::Zero(m,n);
    srand((unsigned)time(NULL));
    for(int h=0;h<ratio*m*n;h++) {
      int i=rand()%m;
      int j=rand()%n;
      d_m(i,j)=(float)(rand()%20000-10000)/1000;
    }
    spMatFloat s_m=d_m.sparseView();
    for(int i=0; i<n; ++i){
        b(i)=2;
    }
    c = s_m * b;
    double start,end,elapse;
    start = getHighResolutionTime();
    for(int i=0; i<ite; ++i){
        c = s_m * b;
    }
    
    end = getHighResolutionTime();
    elapse = end - start;
    
    cout << "m n ratio: " << m << " " << n << " " << ratio << endl;
    cout << "time take: " << elapse/ite << endl;

 
  
 
    return 0;
}