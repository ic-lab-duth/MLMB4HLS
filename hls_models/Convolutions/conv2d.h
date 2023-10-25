#ifndef __DCNN_CONV2D__
#define __DCNN_CONV2D__

#include "defines.h"
#include "ac_math/ac_sigmoid_pwl.h"
#include "mc_scverify.h"

namespace dcnn {

  namespace cpp {

    template<typename dtype_I, typename dtype_O, 
            int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
            int Pt, int Pb, int Pl, int Pr, Activation ACT=linear>
    class Conv2D {
      private:

        typedef weightT       wtype;
        typedef biasT         btype;

        typedef compactDataT<dtype_I, Tn> unrolled_dti;
        typedef compactDataT<dtype_O, Tm> unrolled_dto;

        typedef ndmatrix::Mat3d<dtype_I, R, C, N>  chanI; // Input feature channel datatype
        typedef ndmatrix::Mat3d<dtype_I, R-K+1+ Pt+Pb, C-L+1 +Pl + Pr, M>  chanO; // Output feature channel datatype
        typedef ndmatrix::Mat4d<wtype, M, N, K, L> chanW; // Load weights channel datatype
        typedef ndmatrix::Mat1d<btype, M>          chanB; // Load bias channel datatype
                
        dtype_O activation(dtype_I &x) {
          dtype_O res;
          switch (ACT) {
            case relu: { 
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
            case linear: { 
              res = (dtype_O)x;
              break;
            }
            case sigmoid: {
              #ifdef __USE_FIXED__
              const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in = (x < 0) ? (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)(-x) : (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)x;
              ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp;
              // dtype_O tmp;
              ac_math::ac_sigmoid_pwl(in, tmp);
              //res = ac_math::ac_sigmoid_pwl<dtype_O, AC_TRN, dtype_I>(x);
              const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in1 = 0;
              ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp1;
              ac_math::ac_sigmoid_pwl(in1, tmp1);
              res = (x < 0) ? (dtype_O)(tmp1 - tmp) : (dtype_O)tmp;
              #else
              res = 1/(1+exp(-x));
              #endif
              break;
            }
            default: {
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
          } // end switch

          return res;
        };

      public:
        Conv2D() {};
        ~Conv2D() {};

        void run(chanW &w, chanB &b, chanI &din, chanO &dout) {

          ndmatrix::Mat3d<dtype_I, R, C, M> out_buffer;

          ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the input feature maps
            COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the input feature maps
              IMAP: for (int n = 0; n < N; n+=Tn) {  // foreach channel of the input features
                OMAP: for (int m = 0; m < M; m+=Tm) {  // foreach output feature map
                  _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
                    _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile
                      _TIM: for (int tn = 0; tn < Tn; tn++) { // foreach channel of the tile
                        _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                          
                          if (n+tn == 0) out_buffer[r+tr][c+tc][m+tm] = b[m+tm];
                          
                          dtype_O sum = 0;
                          MADi: for (int k = 0; k < K; k++) {
                            dtype_O inner_sum = 0;
                            bool NPad = (r + tr + k) < Pt;
                            bool SPad = (r + tr + k - Pt) > (R-1);
                            MADj: for (int l = 0; l < L; l++) {
                              bool EPad = (c + tc + l) < Pl;
                              bool WPad = (c + tc + l - Pl) > (C-1);
                              bool isPad = NPad || SPad || EPad || WPad;
                              dtype_I ifeat = (isPad) ? (dtype_I)0 : din[r+tr+k-Pt][c+tc+l-Pl][n+tn]; // Perform on-line ZeroPadding2D
                              inner_sum += ifeat * w[m+tm][n+tn][k][l];
                            }
                            sum += inner_sum;
                          }
                          out_buffer[r+tr][c+tc][m+tm] += sum;

                          if (n+tn == N-1) dout[r+tr][c+tc][m+tm] = activation(out_buffer[r+tr][c+tc][m+tm]);
                        } // _TOM
                      } // _TIM
                    } // _TCS
                  } // _TRS
                } // OMAP                
              } // IMAP
            } // COLS
          } // ROWS
        }
    };

  }; // (namespace) cpp

  namespace hls {
    
    #pragma hls_design
    template<typename dtype_I, typename dtype_O, 
            int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
            int Pt, int Pb, int Pl, int Pr, Activation ACT=linear>
    class Conv2D {
      private:

        typedef dtype_I inner_dt;

        typedef weightT       wtype;
        typedef biasT         btype;

        typedef compactDataT<dtype_I, Tn> unrolled_dti;
        typedef compactDataT<dtype_O, Tm> unrolled_dto;

        typedef ac_channel<unrolled_dti> chanI; // Input feature channel datatype
        typedef ac_channel<unrolled_dto> chanO; // Output feature channel datatype
        typedef ac_channel<bool>         chanL; // Load enable channel datatype
        typedef ac_channel<wtype>        chanW; // Load weights channel datatype
        typedef ac_channel<btype>        chanB; // Load bias channel datatype

        //typedef ndmatrix::Mat2d<unrolled_dti, K, L>         windowbuffer;
        //typedef ndmatrix::Mat4d<wtype, M, N, K, L>                 weightbuffer;
        typedef ndmatrix::Mat1d<dtype_I, Tn*K*L>                     windowbuffer;
        typedef ndmatrix::Mat3d<unrolled_dti, K-1, (C+Pl+Pr)/Tc, Tc> linebuffer;
        typedef ndmatrix::Mat3d<wtype, N/Tn, M/Tm, Tn*Tm*K*L>        weightbuffer;
        typedef ndmatrix::Mat2d<btype, M/Tm, Tm>                     biasbuffer;
        typedef ndmatrix::Mat4d<dtype_O, (R-(K-1)+(Pt+Pb)), (C-(L-1)+(Pl+Pr)), M/Tm, Tm> outbuffer;
        
        //dtype_I window[Tn*K*L];
        windowbuffer window;
        linebuffer   lb;
        outbuffer obuff;
        //weightbuffer weights;
        //wtype weights[N/Tn][M/Tm][Tn*Tm*K*L];
        //biasbuffer   bias;
        
        
        // Updates the values inside the window and line buffers, each time
        // a new set of inputs arrive.
        void updateBuffers(unrolled_dti &pxl, const int c, const int tc) {
          #pragma hls_unroll
          _vert: for (int k = 0; k < K; k++) {
            unrolled_dti lb_out = (k < K-1) ? lb[k][c][tc] : pxl;
            if (k > 0) {
              lb[k-1][c][tc] = lb_out;
            }
            #pragma hls_unroll
            _hori: for (int l = 0; l < L; l++) {
              #pragma hls_unroll
              for (int tn = 0; tn < Tn; tn++) {
                window[tn*K*L + k*L +l] = (l < L-1) ? window[tn*K*L + k*L + l+1] : lb_out[tn];
              }
            } 
          }
        };
        
        // Feeds the K x K calculated products to an addition tree, in order
        // to calculate the partial output feature for a OFM using pixels
        // from a single single IFM
       /* dtype_O madd(const int m, const int n) {
          dtype_O sum = 0;
          MADi: for (int k = 0; k < K; k++) {
            dtype_O inner_sum = 0;
            MADj: for (int l = 0; l < L; l++) {
              inner_sum += window[k][l][n] * weights[m][n][k][l];
            } // MADj
            sum += inner_sum;
          } // MADi
          return sum;
        };*/

        dtype_O activation(dtype_O &x) {
          dtype_O res;
          switch (ACT) {
            case relu: { 
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
            case linear: { 
              res = (dtype_O)x;
              break;
            }
            case sigmoid: {
              // const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in = (x < 0) ? (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)(-x) : (ac_fixed<25,15,false,AC_RND, AC_SAT_SYM>)x;
              // ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp;
              // // dtype_O tmp;
              // ac_math::ac_sigmoid_pwl(in, tmp);
              // // res = ac_math::ac_sigmoid_pwl<dtype_O, AC_TRN, dtype_I>(x);
              // const ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> in1 = 0;
              // ac_fixed<25,15,false,AC_RND, AC_SAT_SYM> tmp1;
              // ac_math::ac_sigmoid_pwl(in1, tmp1);
              // res = (x < 0) ? (dtype_O)(tmp1 - tmp) : (dtype_O)tmp;
                           
              break;
            }
            default: {
              res = (x > 0) ? (dtype_O)x : (dtype_O)0;
              break;
            }
          } // end switch

          return res;
        };

      public:
        Conv2D() {};
        ~Conv2D() {};
        
        #pragma hls_design interface
        void CCS_BLOCK(run)(chanI &din, chanO &dout, /*weightbuffer &weights*/wtype weights[N/Tn][M/Tm][Tn*Tm*K*L], biasbuffer &bias) {

          IMAP: for (int n = 0; n < N/Tn; n++) {  // foreach channel of the input features
            ROWS: for (int r = -Pt; r < R+Pb; r+=Tr) {  // foreach row of the input feature maps
              COLS: for (int c = -Pl; c < C+Pr; c+=Tc) {  // foreach column of the input feature maps

                // Check if this is a border pixel
                bool isPad = ( r < 0) || ( r >= R ) || ( c < 0) || ( c >= C );
                                                                        
                if (din.available(1) || isPad) {  // Here we read the Tn parallel inputs 

                  // Perform on-line ZeroPadding2D
                  unrolled_dti structured_din = (isPad) ? (unrolled_dti)0 : din.read(); 
                  // Update Window and Line buffers
                  updateBuffers(structured_din, c, 0);
            
                  OMAP: for (int m = 0; m < M/Tm; m++) {  // foreach output feature map

                    #pragma hls_unroll  
                    _TIM: for (int tn = 0; tn < Tn; tn++) { // foreach channel of the tile
                      
                      bool write_out = ((n*Tn + tn) == (N-1)) && ( (r + Pt) >= (K-1) ) && ( (c + Pl) >= (L-1) );
                              
                      unrolled_dto structured_dout;
                      #pragma hls_unroll
                      _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                        

                        dtype_O sum = 0;
                        #pragma hls_unroll                      
                        DOT: for (int k = 0; k < K*L; k++) {
                          sum += window[tn*K*L + k] * weights[n][m][tn*Tm*K*L + tm*K*L + k];
                        } // DOT
                        obuff[r][c][m][tm] = (n+tn == 0) ? (dtype_O)(bias[m][tm]+ sum) : (dtype_O)(obuff[r][c][m][tm] + sum);


                        if ( write_out ) {
                          structured_dout[tm] = activation(obuff[r][c][m][tm]);
                          // if (tm == Tm-1) {
                          //   dout.write(structured_dout);
                          //   // std::cout << structured_dout[0] << std::endl;
                          //   // std::cin.get();
                          // }
                        } // (endif) write_out 
                
                      
                      } // _TOM
                      if ( write_out ) {
                        dout.write(structured_dout);
                      }
                    } // _TIM
                  } // OMAP 
                } // endif din.available(1)
              } // ROWS
            } // COLS 
          } // IMAP
        } // (func) run
    };

  }; // (namespace) hls

}; // (namespace) dcnn

#endif
