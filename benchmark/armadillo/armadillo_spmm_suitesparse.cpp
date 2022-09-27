// g++ armadillo_spmm_suitesparse.cpp -o armadillo_spmm_suitesparse -O3 -I ~/armadillo-10.5.1/include -DARMA_DONT_USE_WRAPPER -lblas -llapack -fopenmp
#include <iostream>
#include <armadillo>
#include <sys/time.h>
// #include <omp.h>
using namespace std;
using namespace arma;
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
double getHighResolutionTime(void) {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
  return time_seconds;
}
int main(int argc, char** argv){
  //suitesparse dataset
    string path = "datasets/big.mtx";
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
    int N = 1024;
  
  omp_set_num_threads(threads);
  fmat A(M, K);
  fmat B(K, N);
  
  // prepare sparse matrix data
  srand((unsigned)time(NULL));
  
  while(getline(fp, str)){
        
        nums = split(str, " ");
        int i=atoi(nums[0].c_str());
        int j=atoi(nums[1].c_str());
        float val = atof(nums[2].c_str());
        // cout << str << endl;
        A(i-1,j-1) = val;
        // matrix[(i-1)*cols + (j-1)] = val;
        
    }
  sp_fmat sp_A = sp_fmat(A);
  
  for(int i=0;i<K;i++){
        for(int j=0;j<N;j++){
            B(i,j)=(float)(rand()%20000-10000)/1000;
        }
    }

  double t1 = getHighResolutionTime();
  fmat result(sp_A * sp_A);
  double t2 = getHighResolutionTime();

  cout <<"Take time:" <<t2 - t1 << endl;

  
  return 0;
}
