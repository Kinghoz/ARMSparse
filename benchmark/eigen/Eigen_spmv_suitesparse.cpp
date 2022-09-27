//g++ -I /home/zhengj/eigen-3.3.9 Eigen_suitesparse.cpp -o Eigen_suitesparse -O3 -fopenmp
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <string>
#include <fstream>
#include <vector>
#include <string.h>

using namespace Eigen;
using namespace std;
vector<string> split(const string& str, const string& delim);
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}
typedef Matrix<float,Dynamic,1> DenseVector;
typedef SparseMatrix<float, RowMajor> spMatFloat;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> deMatRowFloat;
typedef Matrix<float, Dynamic, Dynamic, ColMajor> deMatColFloat;
void bench_Sparse(const spMatFloat &m, const deMatColFloat &in, deMatColFloat &o) {
  o=m*in; //o.noalias()=m*in.transpose();
}

void bench_Dense(const deMatRowFloat &m, const deMatRowFloat &in, deMatRowFloat &o) {
  o.noalias()=m*in.transpose();
}

int main(int argc, const char **argv) {
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
    int threads = strtol(argv[1], NULL, 10);
    int ite = 30;
    int M = rows;
    int K = cols;

    deMatRowFloat d_m=deMatRowFloat::Zero(M,K);
    while(getline(fp, str)){
        
        nums = split(str, " ");
        int i=atoi(nums[0].c_str());
        int j=atoi(nums[1].c_str());
        float val = atof(nums[2].c_str());
        // cout << str << endl;
        d_m(i-1,j-1) = val;
        // matrix[(i-1)*cols + (j-1)] = val;
        
    }
    
  Eigen::setNbThreads(threads);
  cout << Eigen::nbThreads() << endl;

  spMatFloat s_m=d_m.sparseView();
  DenseVector b(K), c(M);
  // float ratio=0.1;
  for(int i=0; i<K; ++i){
    b(i)=0.13;
  }
  double take = 0;
  float t_dense=0;
  float t_sparse=0;
  c += s_m * b;
  double t1 = getHighResolutionTime();
    for(int i=0; i<ite; ++i){
        c += s_m * b;
    }
  double t2 = getHighResolutionTime();
  // double sp_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  // t_sparse+=sp_elapsed_secs;
  take = t2 - t1;
  
  
  std::cout << "m k: " << M << " " << K <<endl;
  std::cout<<"time take: "<<take/ite<<std::endl;
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