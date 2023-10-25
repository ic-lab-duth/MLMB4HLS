#ifndef __DCNN_FLATTEN__
#define __DCNN_FLATTEN__

#include "defines.h"
#include "mc_scverify.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype, int R, int C, int N, int Tn, int Tm, int Tr, int Tc>
    class Flatten {
    private:

      typedef ndmatrix::Mat3d<dtype, R, C, N> chanI;
      typedef ndmatrix::Mat1d<dtype, R*C*N>   chanO; 

    public:
      Flatten() {};
      ~Flatten() {};

      void run(chanI &din, chanO &dout) {

        ROWS: for (int r = 0; r < R; r+=Tr) {  
          COLS: for (int c = 0; c < C; c+=Tc) {  
            IMAP: for (int n = 0; n < N; n+=Tn) {
              _TRS: for (int tr = 0; tr < Tr; tr++) { 
                _TCS: for (int tc = 0; tc < Tc; tc++) { 
                  _TIM: for (int tn = 0; tn < Tn; tn++) {

                    dout[(r+tr)*C*N + (c+tc)*N + (n+tn)] = din[r+tr][c+tc][n+tn];

                  } // _TIM
                } // IMAP
              } // _TCS
            } // _TRS
          } // COLS
        } // ROWS
      }; // (function) run


    }; // (class) Flatten

  }; // (namespace) cpp

  namespace hls {

    #pragma hls_design
    template<typename dtype, int R, int C, int N, int Tn, int Tm, int Tr, int Tc>
    class Flatten {
    private:

      typedef compactDataT<dtype, Tn> unrolled_dti;
      typedef compactDataT<dtype, Tm> unrolled_dto;

      typedef ac_channel<unrolled_dti> chanI;
      typedef ac_channel<unrolled_dto> chanO; 

    public:
      Flatten() {};
      ~Flatten() {};

      #pragma hls_design interface
      void CCS_BLOCK(run)(chanI &din, chanO &dout) {

        ndmatrix::Mat1d<dtype, R*C*N> buffer;

        // ~~ READ
        ROWS: for (int r = 0; r < R; r+=Tr) {  
          COLS: for (int c = 0; c < C; c+=Tc) {  
            IMAP: for (int n = 0; n < N; n+=Tn) {
              _TRS: for (int tr = 0; tr < Tr; tr++) { 
                _TCS: for (int tc = 0; tc < Tc; tc++) { 
                  unrolled_dti inp;
                  if (din.available(1)) {
                    inp = din.read();
                    
                    _TIM: for (int tn = 0; tn < Tn; tn++) {
                      buffer[(r+tr)*C*N + (c+tc)*N + (n+tn)] = inp[tn];
                    } // _TIM
                  } // (if) channel is available
                } // IMAP
              } // _TCS
            } // _TRS
          } // COLS
        } // ROWS


        // ~~ WRITE
        OMAP: for (int m = 0; m < R*C*N; m+=Tm) {
          unrolled_dto out;
          _TOM: for (int tm = 0; tm < Tm; tm++) {
            out[tm] = buffer[m+tm];
          } // _TOM
          dout.write(out);
        } // OMAP
      
      }; // (function) run


    }; // (class) Flatten

  }; // (namespace) hls

}; // (namespace) dcnn

#endif
