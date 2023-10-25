#ifndef __ADNNET_POOL2D__
#define __ADNNET_POOL2D__

#include "defines.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype, int R, int C, int M, int K, int L, int Tm, int Tr=1, int Tc=1, Pooling P=Max>
    class Pool2D {
      private:

        
        typedef ndmatrix::Mat3d<dtype, R, C, M>     chanI; 
        typedef ndmatrix::Mat3d<dtype, R/K, C/L, M> chanO; 

        typedef ndmatrix::Mat2d<dtype, K, L>     kernelT;
        typedef ndmatrix::Mat3d<dtype, K, C, Tm>  bufferT;
        typedef ndmatrix::Mat3d<dtype, Tm, K, L> windowT;

        windowT window;
        bufferT buffer;

        dtype max(dtype window[K][L]) {
          dtype max = window[0][0];
          for (int k = 0; k < K; k++)
            for (int l = 0; l < L; l++)
              max = (max > window[k][l]) ? max : window[k][l];
          return max;
        };

        dtype avg(dtype window[K][L]) {
          // TODO: Write function
          return 0;
        };

        dtype f_pool(dtype window[K][L]) {
          dtype out;
          switch (P) {
            case Max: out = max(window); break;
            case Avg: out = avg(window); break;
            default:  out = max(window); break;
          }

          return out;
        }

      public:

        Pool2D() {};
        ~Pool2D() {};

        void run(chanI &inp, chanO &out) {
          
          ROWS: for (int r = 0; r < R; r+=Tr*K) {  // foreach row of the input feature maps
            COLS: for (int c = 0; c < C; c+=Tc*L) {  // foreach column of the input feature maps
              _TRS: for (int tr = 0; tr < Tr; tr+=K) { // foreach row of the tile
                _TCS: for (int tc = 0; tc < Tc; tc+=L) { // foreach column of the tile

                  MAPS: for (int m = 0; m < M; m+=Tm) {
                    for (int tm = 0; tm < Tm; tm++) {
                      
                      dtype window[K][L];
                      // update window
                      for (int k = 0; k < K; k++) {
                        for (int l = 0; l < L; l++) {
                        
                          window[k][l] = inp[r+tr+k][c+tc+l][m+tm];

                        }
                      }
                      out[(r+tr)/K][(c+tc)/L][m+tm] = f_pool(window);
                      // for (int k = 0; k < K; k++) {
                      //   for (int l = 0; l < L; l++) {
                      //     std::cout << window[k][l] << " ";
                      //   }
                      // }
                      // std::cout << std::endl << (r+tr)/K << " " << (c+tc)/L << " " << m+tm << std::endl;
                      // std::cout << out[(r+tr)/K][(c+tc)/L][m+tm] << std::endl;
                    }

                  } // (loop) MAPS
                } // _TCS
              } // _TRS
            } // COLS
          } // ROWS

        }; // (function) compute_layer
    };

  }; // (namespace) cpp

  namespace hls {
    #pragma hls_design
    template<typename dtype, int R, int C, int M, int K, int L, int Tm, int Tr=1, int Tc=1, Pooling P=Max>
    class Pool2D {
      private:

        typedef compactDataT<dtype, Tm> unrolled_dtio;
        
        typedef ac_channel<unrolled_dtio> chanIO; 

        typedef ndmatrix::Mat2d<dtype, K, L>     kernelT;
        typedef ndmatrix::Mat3d<dtype, K, C, Tm>  bufferT;
        typedef ndmatrix::Mat3d<dtype, Tm, K, L> windowT;

        windowT window;
        bufferT buffer;


        dtype max(dtype k1, dtype k2, dtype k3, dtype k4) {
          dtype m1 = (k1 > k2) ? k1 : k2;
          dtype m2 = (k3 > k4) ? k3 : k4;

          return ((m1 > m2) ? m1 : m2);
        };

        dtype avg(dtype k1, dtype k2, dtype k3, dtype k4) {
          // TODO: Write function
          return 0;
        };

        dtype f_pool(dtype k1, dtype k2, dtype k3, dtype k4) {
          dtype out;
          switch (P) {
            case Max:
              out = max(k1, k2, k3, k4);
              break;

            case Avg:
              out = avg(k1, k2, k3, k4);
              break;
            
            default:
              out = max(k1, k2, k3, k4);
              break;
          }

          return out;
        }

      public:

        Pool2D() {};
        ~Pool2D() {};

        #pragma hls_design interface
        void run(chanIO &inp, chanIO &out) {
          
          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the input feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the input feature maps
              _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile

                  MAPS: for (int m = 0; m < M; m+=Tm) {
                    unrolled_dtio prePool;
                    unrolled_dtio postPool;
                    if (inp.available(1)) {
                      prePool = inp.read();

                      for (int tm = 0; tm < Tm; tm++) {

                        // update window
                        // for (int k = 0; k < K; k++) {
                          
                        //   dtype cur_inp = (k < K-1) ? buffer[c+tc][m] : prePool[tm];

                        //   for (int l = 0; l < L; l++) {

                        //   window[tm][k][l+1] = (l < L-1) ? window[tm][k][l] : cur_inp;
                          window[tm][0][0] = window[tm][0][1];
                          window[tm][1][0] = window[tm][1][1];
                          window[tm][0][1] = buffer[0][c+tc][m+tm];
                          window[tm][1][1] = prePool[tm];
                          // update buffer
                          buffer[0][c+tc][m+tm] = prePool[tm];

                        // find max
                        dtype o_feat = f_pool(window[m][0][0], window[m][0][1], window[m][1][0], window[m][1][1]);
                        postPool[tm] = o_feat;
                      }
                      

                      bool val = (((r+tr) % 2) != 0) && (((c+tc) % 2) != 0);
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
