//g++ -I /home/nscc-gz/zhengj/work/eigen-3.3.9 compare.cpp -o compare -O3 -fopenmp
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
using namespace Eigen;

double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}
typedef SparseMatrix<float, RowMajor> spMatFloat;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> deMatRowFloat;
typedef Matrix<float, Dynamic, Dynamic, ColMajor> deMatColFloat;
void bench_Sparse(const spMatFloat &m, const deMatColFloat &in, deMatRowFloat &o) {
  o=m*in; //o.noalias()=m*in.transpose();
}

void bench_Dense(const deMatRowFloat &m, const deMatRowFloat &in, deMatRowFloat &o) {
  o.noalias()=m*in.transpose();
}

int main(int argc, const char **argv) {
  // int m=3000,k=2000,n=4000;
  int m = strtol(argv[1], NULL, 10);
  int k = strtol(argv[2], NULL, 10);
  int n = strtol(argv[3], NULL, 10);
  int threads = strtol(argv[4], NULL, 10);
  Eigen::setNbThreads(threads);
  // printf("%d\n", threads);
  // printf(" %d \n", Eigen::nbThreads());
  float ratio=0.01;
  int iter=1;
  double take = 0;
  float t_dense=0;
  float t_sparse=0;
  // int k = 1000;
  // int m = 1000;
  int sp_size = m*k;
  // int n = 1000; // batch 32 1
  deMatRowFloat d_o1(m,n);
  deMatRowFloat d_o2(m,n);
  
  for(int index=0; index<iter; index++) {
    deMatRowFloat d_m=deMatRowFloat::Zero(m,k);
    deMatColFloat d_b(k,n);
    // std::cout << "yes" << std::endl;
    srand((unsigned)time(NULL));
    for(int h=0;h<ratio*sp_size;h++) {
      int i=rand()%m;
      int j=rand()%k;
      int t =(int)(rand()%10);
      d_m(i,j) = (float )t;
    }
    
    spMatFloat s_m=d_m.sparseView();
    
    for(int i=0;i<k;i++){
      for(int j=0;j<n;j++){
        int t =(int)(rand()%10);
        d_b(i,j) = (float )t;
      }
    }
    
    // {
    //   clock_t begin = clock();
    //   // for(int k=0;k<50;k++) bench_Dense(d_m,d_b,d_o1);
    //   clock_t end = clock();
  	//   double den_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  	//   t_dense+=den_elapsed_secs;
    // }
    {
      int repeat = 1;
      // clock_t begin = clock();
      double t1 = getHighResolutionTime();
      for(int k=0;k<repeat;k++) bench_Sparse(s_m,d_b,d_o2);
      // clock_t end = clock();
      double t2 = getHighResolutionTime();
  	  // double sp_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  	  // t_sparse+=sp_elapsed_secs;
      take = t2 - t1;
    }
  }
  std::cout << "m k n: " << m << " " << k << " "<< n;
  std::cout<<"\tratio\t"<<ratio<<"\tdense\t"<<t_dense/1/iter<<"\tsparse\t"<<take/1/iter<<std::endl;
}
