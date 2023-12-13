#ifndef __MLMB_POOL2D__
#define __MLMB_POOL2D__

#include "defines.h"
#include "utils.h"
#include "mc_scverify.h"

namespace mlmb {

  namespace hls {

    #pragma hls_design
    template<typename dtype, int R, int C, int M, int K, int L, int Tm, int Tr=1, int Tc=1, Pooling P=Max>
    class Pool2D {
      private:

        typedef compactDataT<dtype, Tm> unrolled_dtio;
        
        typedef ac_fixed<16, 4, true> sumtype;
        
        typedef ac_channel<unrolled_dtio> chanIO; 

        typedef ac_int<ac::nbits<R>::val, false> r_indexT;
        typedef ac_int<ac::nbits<C>::val, false> c_indexT;
        typedef ac_int<ac::nbits<K>::val, false> k_indexT;

        typedef ndmatrix::Mat2d<dtype, K, L>       kernelT;
        typedef ndmatrix::Mat2d<kernelT, M/Tm, Tm> windowT;
        typedef ndmatrix::Mat3d<dtype, K, C, Tm>   bufferT;

        typedef ndmatrix::Mat3d<unrolled_dtio, C, M/Tm, K-1> linebuffer;

        linebuffer  lb;

        windowT window;
        bufferT buffer;

         // Updates the values inside the window and line buffers,
        // whenever a new set of inputs arrive.
        void updateBuffers(unrolled_dtio &pxl, int c, int m, k_indexT &lb_idx, bool &compute) {

          if (compute) {
            for (int tm=0; tm < Tm; tm++) {
              for (int k=0; k < K; k++) {
                for (int l=0; l < L-1; l++) {
                  window[m][tm][k][l] = window[m][tm][k][l+1];
                }
                window[m][tm][k][L-1] = (k < K-1) ? lb[c][m][k][tm] : pxl[tm];
              }
            }
          } else {
            lb[c][m][lb_idx] = pxl;
          }

        }; // updateBuffers - end of function


        dtype max(kernelT &win) {
          dtype partial_max[K];

          for (int k=0; k < K; k++) {
            dtype vec[L];
            for (int l=0; l < L; l++)
              vec[l] = win[k][l];

            partial_max[k] = max_tree<L, dtype>(vec);
          }

          return max_tree<K, dtype>(partial_max);
        };

        dtype avg(kernelT &win) {
          sumtype partial_sum[K];

          #pragma hls_unroll
          for (int k=0; k < K; k++) {

            sumtype vec[L];
            for (int l=0; l < L; l++)
              vec[l] = (sumtype)win[k][l];

            sumtype s = add_tree<L, sumtype>(vec);
            switch (L) {
              case 1: 
                partial_sum[k] = (sumtype)s;
                break;
              case 2:
                partial_sum[k] = (sumtype)(s >> 1);
                break;
              case 4:
                partial_sum[k] = (sumtype)(s >> 2);
                break;
              case 8:
                partial_sum[k] = (sumtype)(s >> 3);
                break;
              default:
                partial_sum[k] = (sumtype)s;
                break;
            }
          }

          sumtype total_sum = add_tree<K, sumtype>(partial_sum);
          dtype avgVal;
          switch (K) {
            case 1: 
              avgVal = (dtype)total_sum;
              break;
            case 2:
              avgVal = (dtype)(total_sum >> 1);
              // avgVal = (dtype)(total_sum /2);
              break;
            case 4:
              avgVal = (dtype)(total_sum >> 2);
              // avgVal = (dtype)(total_sum /4);
              break;
            case 8:
              avgVal = (dtype)(total_sum >> 3);
              // avgVal = (dtype)(total_sum /8);
              break;
            default:
              avgVal = (dtype)total_sum;
              break;
          }

          return avgVal;
        };

        dtype f_pool(windowT &win, int m, int tm) {
          dtype out;
          switch (P) {
            case Max:
              out = max(win[m][tm]);
              break;

            case Avg:
              out = avg(win[m][tm]);
              break;
            
            default:
              out = max(win[m][tm]);
              break;
          }

          return out;
        }

      public:

        Pool2D() {};
        ~Pool2D() {};

        #pragma hls_design interface
        void CCS_BLOCK(run)(chanIO &inp, chanIO &out) {
          
          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the input feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the input feature maps
              _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile

                  MAPS: for (int m = 0; m < M/Tm; m++) {
                    unrolled_dtio prePool;
                    unrolled_dtio postPool;
                    #ifndef __SYNTHESIS__
                    if (inp.available(1))
                    #endif 
                    {
                      prePool = inp.read();

                      k_indexT lb_idx;
                      bool compute, compute1;

                      r_indexT rr = (r_indexT)r;
                      c_indexT cc = (c_indexT)c;

                      switch (K)
                      {
                        case 2:
                          lb_idx = rr[0];
                          compute = (lb_idx== 1);
                          compute1 = ( cc[0] == 1 );
                          break;

                        case 4:
                          lb_idx = rr.template slc<2>(0);
                          compute = (lb_idx == 3);
                          compute1 = ( cc.template slc<2>(0) == 3 );
                          break;

                        case 8:
                          lb_idx = rr.template slc<3>(0);
                          compute = (lb_idx == 7);
                          compute1 = ( cc.template slc<3>(0) == 7 );
                          break;
                        
                        default:
                          lb_idx = rr[0];
                          compute = (lb_idx == 1);
                          compute1 = ( cc[0] == 1 );
                          break;
                      }

                      updateBuffers(prePool, c, m, lb_idx, compute);

                      if (compute) {
                        for (int tm=0; tm < Tm; tm++) {
                          dtype o_feat = f_pool(window, m, tm);
                          postPool[tm] = o_feat;
                        }
                      }
                                            

                      bool val = (compute) && (compute1);
                      if (val) {
                        out.write(postPool);
                      }
                    } // (if) channel is available
                  } // (loop) MAPS


                } // _TCS
              } // _TRS
            } // COLS
          } // ROWS

        }; // (function) compute_layer
    };
  }; // (namespace) hls


}; // (namespace) dcnn

#endif 
