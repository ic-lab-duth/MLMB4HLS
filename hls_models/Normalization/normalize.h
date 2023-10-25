#ifndef __ADNNET_NORMALIZE__
#define __ADNNET_NORMALIZE__

#include "defines.h"
#include "mc_scverify.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype, typename otype, int R, int C, int M, int Tm, int Tr=1, int Tc=1>
    class BatchNormalization {
      private:

        typedef scaleT        stype;
        typedef biasT         btype;

        typedef ndmatrix::Mat3d<dtype, R, C, M>  chanI;
        typedef ndmatrix::Mat3d<dtype, R, C, M>  chanO;
        typedef ndmatrix::Mat1d<btype, M>        chanW;
        typedef ndmatrix::Mat1d<btype, M>        chanB;

      public:
        BatchNormalization() {};
        ~BatchNormalization() {};

        void run(chanW &s, chanB &b, chanI &inp, chanO &out){

          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the feature maps
              _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile
                  OMAP: for ( int m = 0; m < M; m+=Tm) {  // foreach feature map
                    _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                        // normOut.itm[tm] = preNorm.itm[tm] * scale[r+tr][c+tc][m+tm] + bias[r+tr][c+tc][m+tm];
                      out[r+tr][c+tc][m+tm] = (inp[r+tr][c+tc][m+tm] * s[m+tm]) + b[m+tm];
                      // std::cout << s[m+tm]  << " " <<  b[m+tm] << std::endl;
                    }
                  }// OMAP
                } // _TCS
              } // _TRS
            } // COLS
          } // ROWS
        }; // run
    };
  }// (namespace) hls

  namespace hls {
    
    #pragma hls_design
    template<typename dtype, typename otype, int R, int C, int M, int Tm, int Tr=1, int Tc=1>
    class BatchNormalization {
      private:

        // typedef unsignedDataT dtype;
        // typedef unsignedDataT otype;
        typedef scaleT        stype;
        typedef biasT         btype;

        typedef compactDataT<dtype, Tm> unrolled_dti;
        typedef compactDataT<otype, Tm> unrolled_dto;

        typedef ac_channel<unrolled_dti> chanI;
        typedef ac_channel<unrolled_dto> chanO;
        typedef ac_channel<bool>     chanL;
        typedef ac_channel<stype>    chanS;
        typedef ac_channel<btype>    chanB;

        // ndmatrix::Mat3d<stype, R, C, M> scale;
        // ndmatrix::Mat3d<btype, R, C, M> bias;
        typedef ndmatrix::Mat1d<stype, M> scale_vec;
        typedef ndmatrix::Mat1d<btype, M> bias_vec;

        scale_vec scale;
        bias_vec  bias;

      public:
        BatchNormalization() {};
        ~BatchNormalization() {};


        #pragma hls_design interface
        void CCS_BLOCK(run)(chanL &l, chanS &s, chanB &b, chanI &inp, chanO &out){

          bool load_enable;
          if (l.available(1)) {
            load_enable = l.read();
          }
          if (load_enable) {
            // rd_SB: for (int r = 0; r < R; r++) {
              // for (int c = 0; c < C; c++) {
                for (int m = 0; m < M; m++) {
                  if (s.available(1)) {
                    scale[m] = s.read();
                    // scale[r][c][m] = s.read();
                  }
                  if (b.available(1)) {
                    bias[m] = b.read();
                    // bias[r][c][m] = b.read();
                  }
                // }
              // }
            }
          }
          else {
            ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the feature maps
              COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the feature maps
                _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                  _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile

                    OMAP: for (int m = 0; m < M; m+=Tm) {  // foreach feature map

                      unrolled_dti preNorm;
                      unrolled_dto normOut;

                      if (inp.available(1)) {
                        preNorm = inp.read();
                        
                        _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                          // normOut.itm[tm] = preNorm.itm[tm] * scale[r+tr][c+tc][m+tm] + bias[r+tr][c+tc][m+tm];
                          normOut.itm[tm] = preNorm.itm[tm] * scale[m+tm] + bias[m+tm];
                        }

                        out.write(normOut);
                      } // endif inp.available(1)
                    }// OMAP

                  } // _TCS
                } // _TRS
              } // COLS
            } // ROWS
          }
        };
    };

  }; // (namespace) hls
} // (namespace) dcnn

#endif
