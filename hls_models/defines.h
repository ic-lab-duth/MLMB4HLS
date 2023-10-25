#ifndef __DCNN_DEFINES__
#define __DCNN_DEFINES__

#include "ac_fixed.h"
#include "fast_float.h"
#include "ndmatrix_matrices.h"
#include "io_datatypes.h"
#include <ac_channel.h>

#define __CPP_MODEL__
// #define __HLS_MODEL__

// #define __SYNTHESIS__

// #define __USE_FIXED__ 
// #define __USE_AC_FLOAT__ 
// #define __USE_FF4HLS__ 

namespace dcnn{

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
    typedef ac_fixed<DATA_W, DATA_I, false, AC_RND, AC_SAT>     unsignedDataT;
    typedef ac_fixed<DATA_W, DATA_I, true,  AC_RND, AC_SAT_SYM> signedDataT;
    typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> weightT;
    typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> scaleT;
    typedef ac_fixed<BIAS_W, BIAS_I, true,  AC_RND, AC_SAT_SYM> biasT;
  #elif defined(__USE_AC_FLOAT__)
    typedef ac::bfloat16 unsignedDataT;
    typedef ac::bfloat16 signedDataT;
    typedef ac::bfloat16 weightT;
  #elif defined(__USE_FF4HLS__)
    typedef ffp16b unsignedDataT;
    typedef ffp16b signedDataT;
    typedef ffp16b weightT;
    typedef ffp16b scaleT;
    typedef ffp16b biasT;
  #else
    typedef float unsignedDataT;
    typedef float signedDataT;
    typedef float weightT;
    typedef float scaleT;
    typedef float biasT;
  #endif
#else 
  typedef ac_fixed<DATA_W, DATA_I, false, AC_RND, AC_SAT>     unsignedDataT;
  typedef ac_fixed<DATA_W, DATA_I, true,  AC_RND, AC_SAT_SYM> signedDataT;
  typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> weightT;
  typedef ac_fixed<WGHT_W, WGHT_I, true,  AC_RND, AC_SAT_SYM> scaleT;
  typedef ac_fixed<BIAS_W, BIAS_I, true,  AC_RND, AC_SAT_SYM> biasT;
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

typedef io_interfaces::INTERFACE Interface_t; 


}; // namespace dcnn

#endif
