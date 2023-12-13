#ifndef __DCNN_DEFINES__
#define __DCNN_DEFINES__

#include "ac_fixed.h"
#include "fast_float.h"
#include "ndmatrix_matrices.h"
// #include "io_datatypes.h"
#include <ac_channel.h>

#define __CPP_MODEL__
// #define __HLS_MODEL__

// #define __SYNTHESIS__

// #define __USE_FIXED__ 
// #define __USE_QUANT__
// #define __USE_AC_FLOAT__ 
// #define __USE_FF4HLS__ 

namespace mlmb{

// define the data precision
#define DATA_W 16
#define DATA_I 5
// define the weight precision
#define WGHT_W 16
#define WGHT_I 5
// define the bias precision
#define BIAS_W 16
#define BIAS_I 5

#ifndef __SYNTHESIS__
  #ifdef __USE_FIXED__
    typedef ac_fixed<DATA_W, DATA_I, false, AC_RND, AC_SAT>     udataT;
    typedef ac_fixed<DATA_W, DATA_I, true,  AC_RND, AC_SAT_SYM> dataT;
    typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> weightT;
    typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> scaleT;
    typedef ac_fixed<BIAS_W, BIAS_I, true,  AC_RND, AC_SAT_SYM> biasT;
    
    // ConvGuard datatypes
    typedef ac_int<DATA_W+13, true>       small_sumT;
    typedef ac_int<2*(DATA_W+13)+5, true> sumT; 
    typedef ac_int<WGHT_W+5, true>        sum_w_T; 


    typedef ac_int<DATA_W, true> checksum_t;

  #elif defined(__USE_AC_FLOAT__)
    typedef ac::bfloat16 udataT;
    typedef ac::bfloat16 dataT;
    typedef ac::bfloat16 weightT;
  #elif defined(__USE_FF4HLS__)
    typedef ffp16b udataT;
    typedef ffp16b dataT;
    typedef ffp16b weightT;
    typedef ffp16b scaleT;
    typedef ffp16b biasT;
  #elif defined(__USE_QUANT__)
    typedef ac_int<8, false>       udataT;
    typedef ac_int<8, true>        dataT;
    typedef ac_int<8, true>        weightT;
    typedef ac_int<8, true>        biasT; 
    typedef ac_int<8, true>        scaleT;

    typedef ac_fixed<16, 8, true>  internalT;

    typedef ac_fixed<16, 0, false> sfactorT; 
    typedef ac_int<8, true>        zeroPointT;
  #else
    typedef float udataT;
    typedef float dataT;
    typedef float weightT;
    typedef float scaleT;
    typedef float biasT;

    typedef float small_sumT;
    typedef float sumT;
    typedef float sum_w_T;

    typedef float checksum_t;
  #endif
#else
  #ifdef __USE_QUANT__
    typedef ac_int<8, false> udataT;
    typedef ac_int<8, true> dataT;
    typedef ac_int<8, true> weightT;
    typedef ac_fixed<16, 0, false> scaleT;
    typedef ac_fixed<16, 0, false> biasT; 
  #else
    typedef ac_fixed<DATA_W, DATA_I, false, AC_RND, AC_SAT>     udataT;
    typedef ac_fixed<DATA_W, DATA_I, true,  AC_RND, AC_SAT_SYM> dataT;
    typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> weightT;
    typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> scaleT;
    typedef ac_fixed<BIAS_W, BIAS_I, true,  AC_RND, AC_SAT_SYM> biasT;
  #endif
#endif

template<typename T, int N>
struct compactDataT {
    T itm[N];  
    
    compactDataT() {};

    compactDataT(T x) {
      #pragma unroll yes
      for (int i=0; i<N; i++) 
        itm[i] = x;  
    };

    T& operator[] (int i) {
      return itm[i];
    }
    
    void operator=(T x) {
      #pragma unroll yes
      for (int i=0; i<N; i++) 
        itm[i] = x;   
    };

    void operator=(T x[N]) {
      #pragma unroll yes
      for (int i=0; i<N; i++) 
        itm[i] = x[i];   
    };
};

typedef enum {relu, linear, softmax, sigmoid} Activation;
typedef enum {Max, Avg}      Pooling;

template<int T, int B, int L, int R>
struct Padding{
  int top = T;
  int bottom = B;
  int left = L;
  int right = R;
};
//typedef struct Padding Padding;

// typedef io_interfaces::INTERFACE Interface_t; 

template<typename T, int Tn, int R, int C, int N>
struct cs_packed
{ 

  compactDataT<T, Tn> data;

  ac_int<ac::nbits<R>::val+1, true> r; 
  ac_int<ac::nbits<C>::val+1, true> c; 
  ac_int<ac::nbits<N>::val+1, true> n; 

  bool last;
};


}; // namespace dcnn

#endif
