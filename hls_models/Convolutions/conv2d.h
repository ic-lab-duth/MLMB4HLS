#ifndef __MLMB_CONV2D__
#define __MLMB_CONV2D__

#include "defines.h"
#include "utils.h"
#include "ac_math/ac_sigmoid_pwl.h"
#include "mc_scverify.h"

namespace mlmb {
  namespace hls {
    
    #pragma hls_design
    template<typename dtype_I, typename dtype_O, 
            int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
            int Pt, int Pb, int Pl, int Pr, Activation ACT, bool SAFE>
    class Conv2D {
      private:


        /* *** Type Definitions *** */

        typedef dtype_I inner_dt;

        typedef weightT       wtype;
        typedef biasT         btype;

        typedef compactDataT<dtype_I, Tn> unrolled_dti;
        typedef compactDataT<dtype_O, Tm> unrolled_dto;

        typedef ac_channel<unrolled_dti> chanI; // Input feature channel datatype
        typedef ac_channel<unrolled_dto> chanO; // Output feature channel datatype

        typedef ndmatrix::Mat4d<unrolled_dti, (C + Pl + Pr) / Tc, Tc, N / Tn, K - 1> linebuffer;
        typedef ndmatrix::Mat3d<dtype_I, N / Tn, Tn, K * L>                          windbuffer;
        typedef ndmatrix::Mat5d<wtype, N / Tn, M / Tm, Tm, Tn, K * L>                weightbuffer;
        typedef ndmatrix::Mat2d<btype, M / Tm, Tm>                                   biasbuffer;



        /* *** Define Static variables of the design *** */

        windbuffer window;
        linebuffer lb;

        ndmatrix::Mat2d<dtype_O, M/Tm, Tm> obuff; 

        /* *** Inline functions definition *** */

        // Updates the values inside the window and line buffers,
        // whenever a new set of inputs arrive.
        void updateBuffers(unrolled_dti &pxl, const int n, const int c, const int tc, ndmatrix::Mat2d<dtype_I, Tn, K*L> &win) {

          #pragma hls_unroll
          _updatebuff_v: for (int k = 0; k < K; k++) {

            unrolled_dti lb_out = (k < K-1) ? lb[c][tc][n][k] : pxl;
            
            if (k > 0) lb[c][tc][n][k-1] = lb_out;
            

            #pragma hls_unroll
            _static_n_0: for (int st_n = 0; st_n < N/Tn; st_n++) {
              if (st_n == n) {

                #pragma hls_unroll
                _wordsize: for (int tn = 0; tn < Tn; tn++) {
                  #pragma hls_unroll
                  _updatebuff_h: for (int l = 0; l < L; l++) {
                  
                    dtype_I tmp = (l < L-1) ? window[st_n][tn][k*L + l+1] : lb_out[tn];
                    window[st_n][tn][k*L + l] = tmp;
                    win[tn][k*L+l] = tmp;
                  } // _updatebuff_h
                } // _wordsize

              } // if
            } // _static_n_0

          } // _updatebuff_v
        }; // updateBuffers - end of function


        dtype_O dot_product(ndmatrix::Mat2d<wtype, Tn, K*L> &wght, ndmatrix::Mat2d<dtype_I, Tn, K*L> &win) {

          dtype_O sum[Tn];
          #pragma hls_unroll
          _TIM: for (int tn = 0; tn < Tn; tn++) { // foreach channel of the tile
            
            dtype_O inner_sum[K*L];
            #pragma hls_unroll                      
            DOT: for (int k = 0; k < K*L; k++) {
              inner_sum[k] = (dtype_O)((dtype_O)(win[tn][k]) * (dtype_O)(wght[tn][k]));
            } // DOT
            sum[tn] = add_tree<(K*L), dtype_O>(inner_sum);
       
          } // _TIM
                            
          return add_tree<Tn, dtype_O>(sum);
        }; // dot_product - end of function
        

        // Inline Activation function. The function is selected through a template parameter
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
              
              #ifdef __USE_FIXED__ 
              ac_fixed<DATA_W,DATA_I,false,AC_RND, AC_SAT_SYM> tmpres;
              ac_math::ac_sigmoid_pwl<AC_TRN, DATA_W, DATA_I, true, AC_RND, AC_SAT_SYM, DATA_W, DATA_I, AC_RND, AC_SAT_SYM>(x, tmpres);
              res = tmpres;
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
        
        #pragma hls_design interface
        void CCS_BLOCK(run)(chanI &din, chanO &dout, weightbuffer &weights, biasbuffer &bias) {

          ROWS: for (int r = -Pt; r < R+Pb; r+=Tr) {  // foreach row of the input feature maps
            COLS: for (int c = -Pl; c < C+Pr; c+=Tc) {  // foreach column of the input feature maps
              
              IMAP: for (int n = 0; n < N/Tn; n++) {  // foreach channel of the input features

                // Check if this is a border pixel
                bool isPad = ( r < 0 ) || ( r >= R ) || ( c < 0 ) || ( c >= C );
                bool valid_comp = ( (r + Pt) >= (K-1) ) && ( (c + Pl) >= (L-1) );
                bool valid_out = valid_comp && (n == (N/Tn-1));
                              
                                                                        
                #ifndef __SYNTHESIS__                                                  
                if (din.available(1))
                #endif 
                {  // Here we read the Tn parallel inputs 

                  // Perform on-line ZeroPadding2D
                  unrolled_dti structured_din = (isPad) ? (unrolled_dti)0 : din.read(); 

                    ndmatrix::Mat2d<dtype_I, Tn, K*L> win; 

                  // Update Window and Line buffers
                  updateBuffers(structured_din, n, c, 0, win);

                  
                  OMAP: for (int m = 0; m < M/Tm; m++) {  // foreach output feature map
                                        

                    unrolled_dto structured_dout;

                    #pragma hls_unroll
                    _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                      
                      ndmatrix::Mat2d<wtype, Tn, K*L> wgt;
                        
                      #pragma hls_unroll
                      _static_n_1: for (int st_n = 0; st_n < N/Tn; st_n++) {
                        if (st_n == n) {

                    
                          for (int tn = 0; tn < Tn; tn++) {
                            for (int k = 0; k < K*L; k++) {
                              wgt[tn][k] = weights[st_n][m][tm][tn][k]; 
                            }
                          }

                        }
                      }
                                                    
                      dtype_O res = dot_product(wgt, win);               
                        

                      dtype_O fin = (n == 0) ? (dtype_O)(bias[m][tm]) : (dtype_O)(obuff[m][tm]);
                      dtype_O final = (dtype_O)(fin + res);

                      if (valid_comp) {
                        obuff[m][tm] = final;

                        structured_dout[tm] = activation(final);
                      }
                    } // _TOM

                    if ( valid_out) {
                      dout.write(structured_dout);
                    }

                    
                  } // OMAP 
                } // endif din.available(1)
              } // IMAP
            } // COLS 
          } // ROWS 
        } // (func) run
    };

    #pragma hls_design
    template<typename dtype_I, typename dtype_O, 
            int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
            int Pt, int Pb, int Pl, int Pr, Activation ACT>
    class Conv2D<dtype_I, dtype_O, R, C, N, M, K, L, Tm, Tn, Tr, Tc, Pt, Pb, Pl, Pr, ACT, true> {
      private:


        /* *** Type Definitions *** */

        typedef dtype_I inner_dt;

        typedef weightT       wtype;
        typedef biasT         btype;
  
        typedef compactDataT<dtype_I, Tn> unrolled_dti;
        typedef compactDataT<dtype_O, Tm> unrolled_dto;

        typedef ac_channel<unrolled_dti> chanI; // Input feature channel datatype
        typedef ac_channel<unrolled_dto> chanO; // Output feature channel datatype
        typedef ac_channel<wtype>        chanW; // Load weights channel datatype
        typedef ac_channel<btype>        chanB; // Load bias channel datatype


        typedef ndmatrix::Mat4d<unrolled_dti, (C + Pl + Pr) / Tc, Tc, N / Tn, K - 1> linebuffer;
        typedef ndmatrix::Mat3d<dtype_I, N / Tn, Tn, K * L>                          windbuffer;
        typedef ndmatrix::Mat5d<wtype, N / Tn, M / Tm, Tm, Tn, K * L>                weightbuffer;
        typedef ndmatrix::Mat2d<btype, M / Tm, Tm>                                   biasbuffer;
        
        typedef checksum_t int_t;
        typedef sumT  csum_t; // Only for ConvGuard
        typedef ac_channel<csum_t> csum_chan;

        typedef cs_packed<dtype_I, Tn, R, C, N> pack2cs_t;
        typedef ac_channel<pack2cs_t> accu_chan;
        
        /* *** Define Static variables of the design *** */

        windbuffer window;
        linebuffer lb;

        ndmatrix::Mat2d<dtype_O, M/Tm, Tm> obuff; 


        /* *** Inline functions definition *** */

        // Updates the values inside the window and line buffers,
        // whenever a new set of inputs arrive.
        void updateBuffers(unrolled_dti &pxl, const int n, const int c, const int tc, ndmatrix::Mat2d<dtype_I, Tn, K*L> &win) {

          #pragma hls_unroll
          _updatebuff_v: for (int k = 0; k < K; k++) {

            unrolled_dti lb_out = (k < K-1) ? lb[c][tc][n][k] : pxl;
            
            if (k > 0) lb[c][tc][n][k-1] = lb_out;
            

            #pragma hls_unroll
            _static_n_0: for (int st_n = 0; st_n < N/Tn; st_n++) {
              if (st_n == n) {

                #pragma hls_unroll
                _wordsize: for (int tn = 0; tn < Tn; tn++) {
                  #pragma hls_unroll
                  _updatebuff_h: for (int l = 0; l < L; l++) {
                  
                    dtype_I tmp = (l < L-1) ? window[st_n][tn][k*L + l+1] : lb_out[tn];
                    window[st_n][tn][k*L + l] = tmp;
                    win[tn][k*L+l] = tmp;
                  } // _updatebuff_h
                } // _wordsize

              } // if
            } // _static_n_0

          } // _updatebuff_v
        }; // updateBuffers - end of function


        dtype_O dot_product(ndmatrix::Mat2d<wtype, Tn, K*L> &wght, ndmatrix::Mat2d<dtype_I, Tn, K*L> &win) {

          dtype_O sum[Tn];
          #pragma hls_unroll
          _TIM: for (int tn = 0; tn < Tn; tn++) { // foreach channel of the tile
            
            dtype_O inner_sum[K*L];
            #pragma hls_unroll                      
            DOT: for (int k = 0; k < K*L; k++) {
              inner_sum[k] = (dtype_O)((dtype_O)(win[tn][k]) * (dtype_O)(wght[tn][k]));
            } // DOT
            sum[tn] = add_tree<(K*L), dtype_O>(inner_sum);
       
          } // _TIM
                            
          return add_tree<Tn, dtype_O>(sum);
        }; // dot_product - end of function
        

        // Inline Activation function. The function is selected through a template parameter
        dtype_O activation(const dtype_O &x) {
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
              ac_fixed<DATA_W,DATA_I,false,AC_RND, AC_SAT_SYM> tmpres;
              ac_math::ac_sigmoid_pwl<AC_TRN, DATA_W, DATA_I, true, AC_RND, AC_SAT_SYM, DATA_W, DATA_I, AC_RND, AC_SAT_SYM>(x, tmpres);
              res = tmpres;
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

        int_t convert_int(dtype_O &fixed_val) {
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

      public:
        Conv2D() {};
        ~Conv2D() {};
        
        #pragma hls_design interface
        void CCS_BLOCK(run)(chanI &din, chanO &dout, weightbuffer &weights, biasbuffer &bias, accu_chan &to_accu, csum_chan &actual) {

          // for (int n = 0; n < N; n++) {
          //   Sx[n] = 0;
          //   #pragma hls_unroll
          //   for (int k = 0; k < K; k++) {
          //     #pragma hls_unroll
          //     for (int l = 0; l < L; l++) {
          //       Sa[n][k][l] = 0;
          //     }
          //   }
          // }

          sumT actualSum = 0;
          // sum_type checksum  = 0;

          ROWS: for (int r = 0; r < R+Pb; r+=Tr) {  // foreach row of the input feature maps
            COLS: for (int c = 0; c < C+Pr; c+=Tc) {  // foreach column of the input feature maps
              
              IMAP: for (int n = 0; n < N/Tn; n++) {  // foreach channel of the input features

                // Check if this is a border pixel
                bool isPad = ( r < 0 ) || ( r >= R ) || ( c < 0 ) || ( c >= C );
                bool valid_comp = ( (r + Pt) >= (K-1) ) && ( (c + Pl) >= (L-1) );
                bool valid_out = valid_comp && (n == (N/Tn-1));
                              
                                                                        
                #ifndef __SYNTHESIS__                                                  
                if (din.available(1))
                #endif  
                {  // Here we read the Tn parallel inputs 

                  // Perform on-line ZeroPadding2D
                  unrolled_dti structured_din = (isPad) ? (unrolled_dti)0 : din.read(); 

                  ndmatrix::Mat2d<dtype_I, Tn, K*L> win; 

                  // Update Window and Line buffers
                  updateBuffers(structured_din, n, c, 0, win);

                  pack2cs_t cs_package;
                  cs_package.data = structured_din;
                  cs_package.r = r;
                  cs_package.c = c;
                  cs_package.n = n;
                  cs_package.last = (r==R-1 && c==C-1 && n==N/Tn-1);
                  to_accu.write(cs_package);
                  
                  OMAP: for (int m = 0; m < M/Tm; m++) {  // foreach output feature map
                                        

                    unrolled_dto structured_dout;

                    #pragma hls_unroll
                    _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                      
                      ndmatrix::Mat2d<wtype, Tn, K*L> wgt;
                        
                      #pragma hls_unroll
                      _static_n_1: for (int st_n = 0; st_n < N/Tn; st_n++) {
                        if (st_n == n) {

                    
                          for (int tn = 0; tn < Tn; tn++) {
                            for (int k = 0; k < K*L; k++) {
                              wgt[tn][k] = weights[st_n][m][tm][tn][k]; 
                            }
                          }

                        }
                      }
                                                    
                      dtype_O res = dot_product(wgt, win);               
                        
                      if (valid_comp) {

                        dtype_O fin = (n == 0) ? (dtype_O)(0) : (dtype_O)(obuff[m][tm]);
                        
                        dtype_O final = (dtype_O)(fin + res);
                        obuff[m][tm] = final;

                        if (valid_out)  {
                          int_t int_val = convert_int(final);
                          actualSum += int_val; // ConvGuard  
                          // actualSum += final; // ConvGuard  
                        }

                        structured_dout[tm] = activation(final+bias[m][tm]);
                      }
                    } // _TOM

                    if ( valid_out) {
                      dout.write(structured_dout);
                    }

                    
                  } // OMAP 
                } // endif din.available(1)

                // if (r == R-1 && c == C-1)  {
                //   #pragma hls_unroll
                //   for (int tn = 0; tn < Tn; tn++) {
                //     #pragma hls_unroll
                //     for (int k = 0; k < K; k++) {
                //       #pragma hls_unroll
                //       for (int l = 0; l < L; l++) {

                //         checksum += (Sx[n*Tn + tn] - Sa[n*Tn + tn][k][l]) * Sw[n*Tn + tn][k][l];
                        
                //       }
                //     }
                //   }
                // }


              } // IMAP
            } // COLS 
          } // ROWS 

          actual.write(actualSum);

          
          // #pragma hls_unroll
          // for (int n = 0; n < N; n++) {
          //   #pragma hls_unroll
          //   for (int k = 0; k < K; k++) {
          //     #pragma hls_unroll
          //     for (int l = 0; l < L; l++) {

          //       checksum += (Sx[n] - Sa[n][k][l]) * Sw[n][k][l];
                
          //     }
          //   }
          // }

          // #ifndef __SYNTHESIS__

          // if (actualSum[2*(DATA_W+13)+5 - 1] == 0) {
          //   while(actualSum[2*(DATA_W+13)+5 - 2] == 0) 
          //     actualSum <<= (1);
          // } else {
          //   while(actualSum[2*(DATA_W+13)+5 - 2] == 1) 
          //     actualSum <<= (1);
          // }

          // if (checksum[2*(DATA_W+13)+5 - 1] == 0) {
          //   while(checksum[2*(DATA_W+13)+5 - 2] == 0) 
          //     checksum <<= (1);
          // } else {
          //    while(checksum[2*(DATA_W+13)+5 - 2] == 1) 
          //     checksum <<= (1);
          // }
          // std::cout << actualSum - checksum << std::endl;
          
          // std::cout << checksum << " " << actualSum << std::endl;
          // #endif

          // return (checksum != actualSum);

        } // (func) run
    };

  }; // (namespace) hls

}; // (namespace) dcnn

#endif
