//g++ -I /home/nscc-gz/zhengj/work/eigen-3.3.9 Eigen_sparseAdd.cpp -o Eigen_sparseAdd -O3 -fopenmp
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
typedef SparseMatrix<int, RowMajor> spMatInt;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> deMatRowInt;
typedef Matrix<int, Dynamic, Dynamic, ColMajor> deMatColInt;


int main(int argc, const char **argv) {
  // int m=3000,k=2000,n=4000;
  int m = strtol(argv[1], NULL, 10);
  int k = strtol(argv[2], NULL, 10);

  int threads = strtol(argv[3], NULL, 10);
  Eigen::setNbThreads(threads);
  // printf("%d\n", threads);
  // printf(" %d \n", Eigen::nbThreads());
  float ratio=0.01;
  int iter=1;
  double take = 0, adjointTime = 0, start;
  float t_dense=0;
  float t_sparse=0;
  // int k = 1000;
  // int m = 1000;
  int sp_size = m*k;
  // int n = 1000; // batch 32 1

    deMatRowInt d_m=deMatRowInt::Zero(m,k);
    deMatRowInt d_m2=deMatRowInt::Zero(m,k);

  for(int index=0; index<iter; index++) {


    // std::cout << "yes" << std::endl;
    srand((unsigned)time(NULL));
    for(int h=0;h<ratio*sp_size;h++) {
      int i=rand()%m;
      int j=rand()%k;
      d_m(i,j)=(int)(rand()%10);
    }
    
    spMatInt s_m=d_m.sparseView();
    spMatInt res = d_m2.sparseView();
      spMatInt res2 = d_m2.sparseView();

    
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
      res = s_m.transpose();
      // clock_t end = clock();
      double t2 = getHighResolutionTime();
  	  // double sp_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  	  // t_sparse+=sp_elapsed_secs;
      take = t2 - t1;
      start = getHighResolutionTime();
      res2 = s_m.adjoint();
      adjointTime = getHighResolutionTime()-start;
//        std::cout << s_m << std::endl;
//        printf("\n");
//        std::cout << res2 << std::endl;
    }
  }
  std::cout << "m k n: " << m << " " << k <<std:: endl;
  std::cout<<"\tratio\t"<<ratio<<"\tdense\t"<<t_dense/1/iter<<"\tsparse\t"<<take/1/iter<<std::endl;
  printf("adjointTime %lf s\n", adjointTime/iter);
}
