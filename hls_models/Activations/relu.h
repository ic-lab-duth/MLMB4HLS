#ifndef __ADNNET_RELU__
#define __ADNNET_RELU__

#include "defines.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype_I, typename dtype_O,
            int R, int C, int M, int Tm=1, int Tr=1, int Tc=1>
    class Relu {
      private:
      
        typedef ndmatrix::Mat3d<dtype_I, R, C, M> chanI;
        typedef ndmatrix::Mat3d<dtype_I, R, C, M> chanO;
        
      public:
        Relu() {};
        ~Relu() {};
        
        void run(chanI &inp, chanO &out){

          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the feature maps
              OMAP: for (int m = 0; m < M; m+=Tm) {  // foreach feature map
                _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                  _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile
                    _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                        
                      dtype_I preAct = inp[r+tr][c+tc][m+tm];
                      dtype_O postAct = (preAct > 0) ? (dtype_O)preAct : (dtype_O)0;
                      out[r+tr][c+tc][m+tm] = postAct;
                      
                    } // _TOM
                  } // _TCS
                } // _TRS
              }// OMAP
            } // COLS
          } // ROWS
        };
    };

  }; // namespace cpp

  namespace hls {

    #pragma hls_design
    template<typename dtype_I, typename dtype_O,
            int R, int C, int M, int Tm=1, int Tr=1, int Tc=1>
    class Relu {
      private:

        typedef compactDataT<dtype_I, Tm> unrolled_dti;
        typedef compactDataT<dtype_O, Tm> unrolled_dto;

        typedef io_interfaces::IO_DATATYPE<unrolled_dti, io_interfaces::CHANNEL> chanI;
        typedef io_interfaces::IO_DATATYPE<unrolled_dto, io_interfaces::CHANNEL> chanO;
        
        
      public:
        Relu() {};
        ~Relu() {};

        #pragma hls_design interface
        void run(chanI &inp, chanO &out){

          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the feature maps
              OMAP: for (int m = 0; m < M; m+=Tm) {  // foreach feature map
                _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                  _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile

                    unrolled_dti preAct;
                    unrolled_dto postAct;

                    if (inp.available(1)) {
                      preAct = inp.read();
                      
                      _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                        postAct[tm] = (preAct[tm] > 0) ? (dtype_O)preAct[tm] : (dtype_O)0;
                      }
                    
                      out.write(postAct);
                    } // endif inp.available(1)

                  } // _TCS
                } // _TRS
              }// OMAP
            } // COLS
          } // ROWS
        };
    };

  }; // (namespace) hls

}; // (namespace) dcnn

#endif
