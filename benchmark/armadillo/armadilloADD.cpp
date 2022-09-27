// g++ armadilloTranspose.cpp -o armadilloTranspose -O3 -I ~/zhengj/work/armadillo-10.5.1/include -DARMA_DONT_USE_WRAPPER -lblas -llapack -fopenmp
#include <iostream>
#include <armadillo>
#include <sys/time.h>
 #include <omp.h>
using namespace std;
using namespace arma;
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}
int main(int argc, char** argv){
  int M = strtol(argv[1], NULL, 10);
  int K = strtol(argv[2], NULL, 10);
  int threads = strtol(argv[3], NULL, 10);
  omp_set_num_threads(threads);
  float ratio = 0.01;
  fmat A(M, K);
  fmat B(M,K);
  
  // prepare sparse matrix data
  srand((unsigned)time(NULL));
  
  for(int h=0; h<ratio*M*K; h++) {
    int i=rand()%M;
    int j=rand()%K;
    A(i,j) = (float)(rand()%20000-10000)/1000;
  }
    for(int h=0; h<ratio*M*K; h++) {
        int i=rand()%M;
        int j=rand()%K;
        B(i,j) = (float)(rand()%20000-10000)/1000;
    }
  sp_fmat sp_A = sp_fmat(A);
    sp_fmat sp_B = sp_fmat(B);

  // fmat B(3,2);
  // fmat A(2,3);
  // A(0,0)=1,A(0,1)=0,A(0,2)=2,A(1,0)=0,A(1,1)=3, A(1,2)=4;
  // B(0,0)=1,B(0,1)=2,B(1,0)=2,B(1,1)=1,B(2,0)=1,B(2,1)=1;
  // sp_fmat sp_A = sp_fmat(A);
  double t1 = getHighResolutionTime();
  sp_fmat C = sp_A + sp_B;
  double t2 = getHighResolutionTime();
//    fmat result(sp_A * B);
  cout <<"Take time:" <<t2 - t1 << endl;
  // cout << "sp_A " << endl;
  // cout << sp_A << endl;
  // cout << "B" <<endl;
  // cout << B << endl;
  // cout << "result" << endl;
  // cout << result << endl;
  
  return 0;
}
