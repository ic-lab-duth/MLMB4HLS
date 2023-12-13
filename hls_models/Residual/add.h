#ifndef __MLMB_ADD__
#define __MLMB_ADD__

#include "defines.h"
#include "utils.h"
#include "mc_scverify.h"

namespace mlmb {
namespace hls {

#pragma hls_design
template<typename dtype_I, typename dtype_O, int R, int C, int N, int Tn>
class Add {
private:

  typedef compactDataT<dtype_I, Tn> unrolled_dti;

  typedef ac_channel<unrolled_dti> chanI;
  typedef ac_channel<unrolled_dto> chanO;


  typedef ndmatrix::Mat3d<dtype_I, R, C, N> mapBuffer;

public:
  Add() {};
  ~Add() {};

  #pragma hls_design interface
  void CCS_BLOCK(run)(chanI &din, chanO &dout, mapBiffer &residual) {

    for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
        for (int n = 0; n < N/Tn; n++) {

          #ifndef __SYNTHESIS__
          if (din.available(1))
          #endif
          {

            unrolled_dti inp = din.read();
            unrolled_dto out;
            #pragma hls_unroll
            for (int tn = 0; tn < Tn; tn++) {
              out[tn] = inp[tn] + residual[r][c][n*Tn+tn];
            }

            dout.write(out);

          }
        }
      }
    }
  }


};

}
}
#endif
