#ifndef __MLMB_SCONV2D__
#define __MLMB_SCONV2D__

#include "defines.h"
#include "utils.h"
#include "ac_math/ac_sigmoid_pwl.h"
#include "mc_scverify.h"

namespace mlmb {
  namespace hls {
    
    #pragma hls_design
    template<typename dtype_I, typename dtype_O, 
            int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
            int Pt, int Pb, int Pl, int Pr, int Sr, int Sc, Activation ACT, bool SAFE>
    class SConv2D {
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
        void updateBuffers(unrolled_dti &pxl, int n, int c, int tc, ndmatrix::Mat2d<dtype_I, Tn, K*L> &win) {

          #pragma hls_unroll
          _updatebuff_v: for (int k = 0; k < K+Sr-1; k++) {

            unrolled_dti lb_out = (k < K-1) ? lb[c][tc][n][k] : pxl;

            if ( k >= Sr && active_lb[k-Sr])  lb[c][tc][n][k-Sr] = lb_out;

            if (k < K) {

              #pragma hls_unroll
              _static_n_0: for (int st_n = 0; st_n < N/Tn; st_n++) {
                if (st_n == n) {

                  #pragma hls_unroll
                  _wordsize: for (int tn = 0; tn < Tn; tn++) {
                    #pragma hls_unroll
                    _updatebuff_h: for (int l = 0; l < L; l++) {
                      if (shift_data[l]) {
                        dtype_I tmp = (l < L-Sc) ? window[st_n][tn][k*L + l+Sc] : lb_out[tn];
                        window[st_n][tn][k*L + l] = tmp;
                      }
                      // win[tn][k*L+l] = tmp;
                    } // _updatebuff_h
                  } // _wordsize

                } // if
              } // _static_n_0
            }
          } // _updatebuff_v
          for (int k=0; k < K; k++ ) 
            for (int l=0; l < L; l++) 
              for (int tn=0; tn < Tn; tn++) 
                win[tn][k*L+l] = window[n][tn][k*L + l];

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

        ndmatrix::Mat1d<bool, K-1> active_lb;
        ndmatrix::Mat1d<bool, K> shift_data;

        typedef ac_int<ac::nbits< (((C+Pl+Pr) > (R+Pt+Pb)) ? (C+Pl+Pr) : (R+Pt+Pb)) >::val, false> feature_iT;
        typedef ac_int<ac::nbits<((K > L) ? K : L)>::val, false> kernel_iT;
        typedef ac_int<3, false> strideT;

        bool decode(feature_iT input_i, kernel_iT curr_i, strideT MOD) {

          bool equalMOD;
          switch (MOD) {
            case (1): 
              equalMOD = true;
              break; 
            
            case (2): 
              equalMOD = (input_i[0] == curr_i[0]);
              break;
            
            case (4): 
              equalMOD = ((input_i[0] == curr_i[0]) && (input_i[1] == curr_i[1]));
              break;
            
            default: 
              strideT cond1 = input_i % MOD;
              strideT cond2 = curr_i % MOD;

              equalMOD = (cond1 == cond2);
              break;
            
          }
          return equalMOD;
        };


      public:
        SConv2D() {};
        ~SConv2D() {};
        
        #pragma hls_design interface
        void CCS_BLOCK(run)(chanI &din, chanO &dout, weightbuffer &weights, biasbuffer &bias) {

          ROWS: for (int r = -Pt; r < R+Pb; r+=Tr) {  // foreach row of the input feature maps
            feature_iT pr = r+Pt;
            
            COLS: for (int c = -Pl; c < C+Pr; c+=Tc) {  // foreach column of the input feature maps
              feature_iT pc = c+Pl;

              #pragma hls_unroll
              LB_DECODE: for (int k=0; k < K-1; k++) {
                kernel_iT pk = k;
                active_lb[k] = decode(pr, pk, Sr);
              }


              const kernel_iT pK = K-1;
              bool active_row = decode(pr, pK, Sr) && (pr >= K-1);
              
              #pragma hls_unroll
              WB_DECODE: for (int l=0; l < L; l++) {
                kernel_iT pl = l;
                shift_data[l] = decode(pc, pl, Sc) && active_row;
              }

              // std::cout << pr << ":{" << active_lb[0] << ", " << active_lb[1] << "}, ";
              // std::cout << pc << ":{" << shift_data[0] << ", " << shift_data[1] << ", " << shift_data[2] << "}" << std::endl;
              
              IMAP: for (int n = 0; n < N/Tn; n++) {  // foreach channel of the input features

                // Check if this is a border pixel
                bool isPad = ( r < 0 ) || ( r >= R ) || ( c < 0 ) || ( c >= C );
                bool valid_comp = ( pr >= (K-1) ) && ( pc >= (L-1) );
                bool valid_out = valid_comp && (n == (N/Tn-1));
                
                                                                        
                #ifndef __SYNTHESIS__                                                  
                if (din.available(1))
                #endif 
                {  // Here we read the Tn parallel inputs 

                  // Perform on-line ZeroPadding2D
                  unrolled_dti structured_din = (isPad) ? (unrolled_dti)0 : din.read(); 

                  ndmatrix::Mat2d<dtype_I, Tn, K*L> win; 

                  // Update Window and Line buffers
                  updateBuffers(structured_din, n, pc, 0, win);

                  // for (int k=0; k < K; k++ ) {
                  //   for (int l=0; l < L; l++) {
                  //     if (pr == 24 && n == 4) {
                  //       std::cout << win[0][k*L+l] << " ";
                  //     }
                  //   }
                  //   if (pr == 24 && n == 4) {
                  //     std::cout << std::endl;
                  //   }
                  // }

                  
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

                      if (shift_data[K-1] && valid_comp) {                        
                        dtype_O fin = (n == 0) ? (dtype_O)(bias[m][tm]) : (dtype_O)(obuff[m][tm]);
                        dtype_O res = dot_product(wgt, win);               
                          
                        dtype_O final = (dtype_O)(fin + res);

                        obuff[m][tm] = final;
                        if (n == 0 && pr == 28 && pc == 8 && m == 4) {
                          std::cout << fin << " " << obuff[m][0] << std::endl;
                        }

                        structured_dout[tm] = activation(final);
                      }
                    } // _TOM

                    if ( valid_out && shift_data[K-1]) {
                      // std::cout << r+Pl << " " << c+Pt << " " << m << std::endl;
                      dout.write(structured_dout);
                    }

                    
                  } // OMAP 
                } // endif din.available(1)
              } // IMAP
            } // COLS 
          } // ROWS 
        } // (func) run

    };

  //   #pragma hls_design
  //   template<typename dtype_I, typename dtype_O, 
  //           int R, int C, int N, int M, int K, int L, int Tm, int Tn, int Tr, int Tc, 
  //           int Pt, int Pb, int Pl, int Pr, Activation ACT>
  //   class Conv2D<dtype_I, dtype_O, R, C, N, M, K, L, Tm, Tn, Tr, Tc, Pt, Pb, Pl, Pr, ACT, true> {
  //     private:


  //       /* *** Type Definitions *** */

  //       typedef dtype_I inner_dt;

  //       typedef weightT       wtype;
  //       typedef biasT         btype;
  
  //       typedef compactDataT<dtype_I, Tn> unrolled_dti;
  //       typedef compactDataT<dtype_O, Tm> unrolled_dto;

  //       typedef ac_channel<unrolled_dti> chanI; // Input feature channel datatype
  //       typedef ac_channel<unrolled_dto> chanO; // Output feature channel datatype
  //       typedef ac_channel<wtype>        chanW; // Load weights channel datatype
  //       typedef ac_channel<btype>        chanB; // Load bias channel datatype


  //       typedef ndmatrix::Mat4d<unrolled_dti, (C + Pl + Pr) / Tc, Tc, N / Tn, K - 1> linebuffer;
  //       typedef ndmatrix::Mat3d<dtype_I, N / Tn, Tn, K * L>                          windbuffer;
  //       typedef ndmatrix::Mat5d<wtype, N / Tn, M / Tm, Tm, Tn, K * L>                weightbuffer;
  //       typedef ndmatrix::Mat2d<btype, M / Tm, Tm>                                   biasbuffer;
        
  //       typedef checksum_t int_t;
  //       typedef sumT  csum_t; // Only for ConvGuard
  //       typedef ac_channel<csum_t> csum_chan;

  //       typedef cs_packed<dtype_I, Tn, R, C, N> pack2cs_t;
  //       typedef ac_channel<pack2cs_t> accu_chan;
        
  //       /* *** Define Static variables of the design *** */

  //       windbuffer window;
  //       linebuffer lb;

  //       ndmatrix::Mat2d<dtype_O, M/Tm, Tm> obuff; 


  //       /* *** Inline functions definition *** */

  //       // Updates the values inside the window and line buffers,
  //       // whenever a new set of inputs arrive.
  //       void updateBuffers(unrolled_dti &pxl, const int n, const int c, const int tc, ndmatrix::Mat2d<dtype_I, Tn, K*L> &win) {

  //         #pragma hls_unroll
  //         _updatebuff_v: for (int k = 0; k < K; k++) {

  //           unrolled_dti lb_out = (k < K-1) ? lb[c][tc][n][k] : pxl;
            
  //           if (k > 0) lb[c][tc][n][k-1] = lb_out;
            

  //           #pragma hls_unroll
  //           _static_n_0: for (int st_n = 0; st_n < N/Tn; st_n++) {
  //             if (st_n == n) {

  //               #pragma hls_unroll
  //               _wordsize: for (int tn = 0; tn < Tn; tn++) {
  //                 #pragma hls_unroll
  //                 _updatebuff_h: for (int l = 0; l < L; l++) {
                  
  //                   dtype_I tmp = (l < L-1) ? window[st_n][tn][k*L + l+1] : lb_out[tn];
  //                   window[st_n][tn][k*L + l] = tmp;
  //                   win[tn][k*L+l] = tmp;
  //                 } // _updatebuff_h
  //               } // _wordsize

  //             } // if
  //           } // _static_n_0

  //         } // _updatebuff_v
  //       }; // updateBuffers - end of function


  //       dtype_O dot_product(ndmatrix::Mat2d<wtype, Tn, K*L> &wght, ndmatrix::Mat2d<dtype_I, Tn, K*L> &win) {

  //         dtype_O sum[Tn];
  //         #pragma hls_unroll
  //         _TIM: for (int tn = 0; tn < Tn; tn++) { // foreach channel of the tile
            
  //           dtype_O inner_sum[K*L];
  //           #pragma hls_unroll                      
  //           DOT: for (int k = 0; k < K*L; k++) {
  //             inner_sum[k] = (dtype_O)((dtype_O)(win[tn][k]) * (dtype_O)(wght[tn][k]));
  //           } // DOT
  //           sum[tn] = add_tree<(K*L), dtype_O>(inner_sum);
       
  //         } // _TIM
                            
  //         return add_tree<Tn, dtype_O>(sum);
  //       }; // dot_product - end of function
        

  //       // Inline Activation function. The function is selected through a template parameter
  //       dtype_O activation(const dtype_O &x) {
  //         dtype_O res;
  //         switch (ACT) {

  //           case relu: { 
  //             res = (x > 0) ? (dtype_O)x : (dtype_O)0;
  //             break;
  //           }

  //           case linear: { 
  //             res = (dtype_O)x;
  //             break;
  //           }

  //           case sigmoid: {
              
  //             #ifdef __USE_FIXED__ 
  //             ac_fixed<DATA_W,DATA_I,false,AC_RND, AC_SAT_SYM> tmpres;
  //             ac_math::ac_sigmoid_pwl<AC_TRN, DATA_W, DATA_I, true, AC_RND, AC_SAT_SYM, DATA_W, DATA_I, AC_RND, AC_SAT_SYM>(x, tmpres);
  //             res = tmpres;
  //             #else
  //             res = 1/(1+exp(-x));     
  //             #endif        
  //             break;
  //           }

  //           default: {
  //             res = (x > 0) ? (dtype_O)x : (dtype_O)0;
  //             break;
  //           }
  //         } // end switch

  //         return res;
  //       };

  //       int_t convert_int(dtype_O &fixed_val) {
  //       int_t int_val;

  //       #ifdef __USE_FIXED__
  //       #pragma hls_unroll
  //       FIX2INT: for (int dw=0; dw < DATA_W; dw++) {
  //         int_val[dw] = fixed_val[dw];
  //       }
  //       #else
  //       int_val = fixed_val;
  //       #endif

  //       return int_val;
  //     };

  //     public:
  //       Conv2D() {};
  //       ~Conv2D() {};
        
  //       #pragma hls_design interface
  //       void CCS_BLOCK(run)(chanI &din, chanO &dout, weightbuffer &weights, biasbuffer &bias, accu_chan &to_accu, csum_chan &actual) {

  //         // for (int n = 0; n < N; n++) {
  //         //   Sx[n] = 0;
  //         //   #pragma hls_unroll
  //         //   for (int k = 0; k < K; k++) {
  //         //     #pragma hls_unroll
  //         //     for (int l = 0; l < L; l++) {
  //         //       Sa[n][k][l] = 0;
  //         //     }
  //         //   }
  //         // }

  //         sumT actualSum = 0;
  //         // sum_type checksum  = 0;

  //         ROWS: for (int r = 0; r < R+Pb; r+=Tr) {  // foreach row of the input feature maps
  //           COLS: for (int c = 0; c < C+Pr; c+=Tc) {  // foreach column of the input feature maps
              
  //             IMAP: for (int n = 0; n < N/Tn; n++) {  // foreach channel of the input features

  //               // Check if this is a border pixel
  //               bool isPad = ( r < 0 ) || ( r >= R ) || ( c < 0 ) || ( c >= C );
  //               bool valid_comp = ( (r + Pt) >= (K-1) ) && ( (c + Pl) >= (L-1) );
  //               bool valid_out = valid_comp && (n == (N/Tn-1));
                              
                                                                        
  //               #ifndef __SYNTHESIS__                                                  
  //               if (din.available(1))
  //               #endif  
  //               {  // Here we read the Tn parallel inputs 

  //                 // Perform on-line ZeroPadding2D
  //                 unrolled_dti structured_din = (isPad) ? (unrolled_dti)0 : din.read(); 

  //                 ndmatrix::Mat2d<dtype_I, Tn, K*L> win; 

  //                 // Update Window and Line buffers
  //                 updateBuffers(structured_din, n, c, 0, win);

  //                 pack2cs_t cs_package;
  //                 cs_package.data = structured_din;
  //                 cs_package.r = r;
  //                 cs_package.c = c;
  //                 cs_package.n = n;
  //                 cs_package.last = (r==R-1 && c==C-1 && n==N/Tn-1);
  //                 to_accu.write(cs_package);
                  
  //                 OMAP: for (int m = 0; m < M/Tm; m++) {  // foreach output feature map
                                        

  //                   unrolled_dto structured_dout;

  //                   #pragma hls_unroll
  //                   _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                      
  //                     ndmatrix::Mat2d<wtype, Tn, K*L> wgt;
                        
  //                     #pragma hls_unroll
  //                     _static_n_1: for (int st_n = 0; st_n < N/Tn; st_n++) {
  //                       if (st_n == n) {

                    
  //                         for (int tn = 0; tn < Tn; tn++) {
  //                           for (int k = 0; k < K*L; k++) {
  //                             wgt[tn][k] = weights[st_n][m][tm][tn][k]; 
  //                           }
  //                         }

  //                       }
  //                     }
                                                    
  //                     dtype_O res = dot_product(wgt, win);               
                        
  //                     if (valid_comp) {

  //                       dtype_O fin = (n == 0) ? (dtype_O)(0) : (dtype_O)(obuff[m][tm]);
                        
  //                       dtype_O final = (dtype_O)(fin + res);
  //                       obuff[m][tm] = final;

  //                       if (valid_out)  {
  //                         int_t int_val = convert_int(final);
  //                         actualSum += int_val; // ConvGuard  
  //                         // actualSum += final; // ConvGuard  
  //                       }

  //                       structured_dout[tm] = activation(final+bias[m][tm]);
  //                     }
  //                   } // _TOM

  //                   if ( valid_out) {
  //                     dout.write(structured_dout);
  //                   }

                    
  //                 } // OMAP 
  //               } // endif din.available(1)

  //               // if (r == R-1 && c == C-1)  {
  //               //   #pragma hls_unroll
  //               //   for (int tn = 0; tn < Tn; tn++) {
  //               //     #pragma hls_unroll
  //               //     for (int k = 0; k < K; k++) {
  //               //       #pragma hls_unroll
  //               //       for (int l = 0; l < L; l++) {

  //               //         checksum += (Sx[n*Tn + tn] - Sa[n*Tn + tn][k][l]) * Sw[n*Tn + tn][k][l];
                        
  //               //       }
  //               //     }
  //               //   }
  //               // }


  //             } // IMAP
  //           } // COLS 
  //         } // ROWS 

  //         actual.write(actualSum);

          
  //         // #pragma hls_unroll
  //         // for (int n = 0; n < N; n++) {
  //         //   #pragma hls_unroll
  //         //   for (int k = 0; k < K; k++) {
  //         //     #pragma hls_unroll
  //         //     for (int l = 0; l < L; l++) {

  //         //       checksum += (Sx[n] - Sa[n][k][l]) * Sw[n][k][l];
                
  //         //     }
  //         //   }
  //         // }

  //         // #ifndef __SYNTHESIS__

  //         // if (actualSum[2*(DATA_W+13)+5 - 1] == 0) {
  //         //   while(actualSum[2*(DATA_W+13)+5 - 2] == 0) 
  //         //     actualSum <<= (1);
  //         // } else {
  //         //   while(actualSum[2*(DATA_W+13)+5 - 2] == 1) 
  //         //     actualSum <<= (1);
  //         // }

  //         // if (checksum[2*(DATA_W+13)+5 - 1] == 0) {
  //         //   while(checksum[2*(DATA_W+13)+5 - 2] == 0) 
  //         //     checksum <<= (1);
  //         // } else {
  //         //    while(checksum[2*(DATA_W+13)+5 - 2] == 1) 
  //         //     checksum <<= (1);
  //         // }
  //         // std::cout << actualSum - checksum << std::endl;
          
  //         // std::cout << checksum << " " << actualSum << std::endl;
  //         // #endif

  //         // return (checksum != actualSum);

  //       } // (func) run
  //   };

  }; // (namespace) hls

}; // (namespace) dcnn

#endif
