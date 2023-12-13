#ifndef __MLMB_CONVGUARD__
#define __MLMB_CONVGUARD__
#include "defines.h"
#include "mc_scverify.h"


namespace mlmb {
namespace hls {

template<typename T, int Tn, int R, int C, int N, int K, int L>
class ConvGuard {
private:

  typedef cs_packed<T, Tn, R, C, N> packed_t;
  typedef compactDataT<T, Tn> unrolled_dti;

  typedef sumT       csum_t;
  typedef small_sumT accu_t;
  typedef sum_w_T    wsum_t;


  typedef ac_channel<packed_t> chan_in;
  typedef ac_channel<csum_t>   csum_in;

  typedef checksum_t  int_t;
  typedef dataT fix_t;
  
  typedef ac_int<ac::nbits<R>::val+1,true> index_r;
  typedef ac_int<ac::nbits<C>::val+1,true> index_c;
  typedef ac_int<ac::nbits<N>::val+1,true> index_n;

  #ifdef __USE_FIXED__
  typedef ac_int<(DATA_W+13)*K*L, false> acc_rf_t;
  #else
  typedef accu_t acc_rf_t[K*L];
  #endif

  typedef ndmatrix::Mat3d<wsum_t, N, K, L> sumWbuffer;

  ndmatrix::Mat2d<accu_t, N/Tn, Tn> Sx;
  ndmatrix::Mat2d<acc_rf_t, N/Tn, Tn> Sa;

  int_t convert_int(fix_t &fixed_val) {
    int_t int_val;

    #ifdef __USE_FIXED__
    #pragma hls_unroll
    FIX2INT: for (int dw=0; dw < DATA_W; dw++) {
      int_val[dw] = fixed_val[dw];
    }
    #else
    int_val = fixed_val;
    #endif

    return int_val;
  };

  #ifdef __USE_FIXED__
  acc_rf_t accumulate_slc(acc_rf_t word, int_t &add_val, bool do_acc[K*L]) 
  #else
  void accumulate_slc(acc_rf_t &word, int_t &add_val, bool do_acc[K*L]) 
  #endif
  {
    acc_rf_t new_word;

    #pragma hls_unroll
    ACC: for (int i=0; i < K*L; i++) {
      #ifdef __USE_FIXED__
      accu_t slice = word.template slc<DATA_W+13>(i*(DATA_W+13));
      #else
      accu_t slice = word[i];
      #endif

      slice += (do_acc[i]) ? add_val : (int_t)0;

      #ifdef __USE_FIXED__
      new_word.set_slc(i*(DATA_W+13), slice);
      #else
      new_word[i] = slice;
      word[i] = new_word[i];
      #endif
    }

    #ifdef __USE_FIXED__
    return new_word;
    #endif
  }


  void cg_accumulate(unrolled_dti &inp, index_r &r, index_c &c, index_n &n) {

    int_t int_val[Tn];

    #pragma hls_unroll
    CONVERT: for (int tn=0; tn < Tn; tn++) {
      int_val[tn] = convert_int(inp[tn]);
    }

    #pragma hls_unroll
    ACC_SX: for (int tn=0; tn < Tn; tn++) {
      Sx[n][tn] += int_val[tn];
    }

    bool do_acc[K*L];
    #pragma hls_unroll
    DECODE_K: for (int k=0; k < K; k++) {
      bool partial = (k>r || r>R-K+k);
      #pragma hls_unroll
      DECODE_L: for (int l=0; l < L; l++) {
        do_acc[k*L+l] = partial || (l>c || c>C-L+l);
      }   
    }

    #pragma hls_unroll
    ACCU_SA: for (int tn=0; tn < Tn; tn++) {
      #ifdef __USE_FIXED__
      Sa[n][tn] = accumulate_slc(Sa[n][tn], int_val[tn], do_acc);
      #else
      accumulate_slc(Sa[n][tn], int_val[tn], do_acc);
      #endif
    }
          
  };

  csum_t checksum_calc(sumWbuffer &Sw) {

    csum_t checksum = 0;

    CS_MAC: 
    for (int n = 0; n < N/Tn; n++) {
      for (int tn=0; tn < Tn; tn++) {
        for (int k = 0; k < K; k++) {
          for (int l = 0; l < L; l++) {
            #ifdef __USE_FIXED__
            checksum += (Sx[n][tn] - Sa[n][tn].template slc<DATA_W+13>((k*L+l)*(DATA_W+13))) * Sw[n*Tn+tn][k][l];
            #else
            checksum += (Sx[n][tn] - Sa[n][tn][k*L+l]) * Sw[n*Tn+tn][k][l];
            #endif
          }
        }
      }
    }

    return checksum;
  };

public:

  ConvGuard() {
    #ifndef __SYNTHESIS__
    #ifdef __USE_FIXED__
    ac::init_array<AC_VAL_0>(&Sx[0][0], N);
    ac::init_array<AC_VAL_0>(&Sa[0][0], N);
    #else
    for (int i=0; i < N/Tn; i++) {
      for (int t=0; t<Tn; t++) {
        Sx[i][t] = 0;
        for (int j=0; j < K; j++) {
          for (int k=0; k < L; k++) {
            Sa[i][t][j*L+k] = 0;
          }
        }
      }
    }

    #endif
    #else
    ac::init_array<AC_VAL_0>(&Sx[0][0], N);
    ac::init_array<AC_VAL_0>(&Sa[0][0], N);
    #endif
  };
  ~ConvGuard(){};

  bool CCS_BLOCK(run)(chan_in &features, csum_in &actual, sumWbuffer &Sw ) {

     #ifndef __SYNTHESIS__
    #ifdef __USE_FIXED__
    ac::init_array<AC_VAL_0>(&Sx[0][0], N);
    ac::init_array<AC_VAL_0>(&Sa[0][0], N);
    #else
    for (int i=0; i < N/Tn; i++) {
      for (int t=0; t<Tn; t++) {
        Sx[i][t] = 0;
        for (int j=0; j < K; j++) {
          for (int k=0; k < L; k++) {
            Sa[i][t][j*L+k] = 0;
          }
        }
      }
    }

    #endif
    #else
    ac::init_array<AC_VAL_0>(&Sx[0][0], N);
    ac::init_array<AC_VAL_0>(&Sa[0][0], N);
    #endif
    
    for (index_r r =0; r<R; r++) {
      for (index_c c=0; c<C; c++) {
        for (index_n n=0; n<N/Tn; n++) {
          #ifndef __SYNTHESIS__
          if (features.available(1)) {
          #endif  

            packed_t din = features.read();

            cg_accumulate(din.data, r, c, n);

          #ifndef __SYNTHESIS__
          }
          #endif
        }
      }
    }
    bool equal;
    csum_t checksum = checksum_calc(Sw);
    std::cout << checksum << std::endl;
    
    #ifndef __SYNTHESIS__
    if (actual.available(1)) {
    #endif
      csum_t actualsum = actual.read();

      equal = (checksum == actualsum);

      #ifndef __SYNTHESIS__
      #ifdef __USE_FIXED__
      ac::init_array<AC_VAL_0>(&Sx[0][0], N);
      ac::init_array<AC_VAL_0>(&Sa[0][0], N);

      bool print=false;
      for (int dw=0; dw < 2*(DATA_W+13)+5; dw++) {
        if (!print && checksum[dw])
          print=true;
        if (print)
          std::cout << checksum[dw];
      }
      std::cout << std::endl;
      
      print=false;
      for (int dw=0; dw < 2*(DATA_W+13)+5; dw++) {
        if (!print && actualsum[dw])
          print=true;
        if (print)
        std::cout << actualsum[dw];
      }
      std::cout << std::endl;

      

      #else
      for (int i=0; i < N/Tn; i++) {
        for (int t=0; t<Tn; t++) {
          Sx[i][t] = 0;
          for (int j=0; j < K; j++) {
            for (int k=0; k < L; k++) {
              Sa[i][t][j*L+k] = 0;
            }
          }
        }
      }
      std::cout << checksum << " " << actualsum << std::endl;
      #endif
      
      #else
      ac::init_array<AC_VAL_0>(&Sx[0][0], N);
      ac::init_array<AC_VAL_0>(&Sa[0][0], N);
      #endif


    #ifndef __SYNTHESIS__
    }
    #endif

    
    
    return equal;
  }

};

} // namespace hls
} // namespace dcnn

#endif
