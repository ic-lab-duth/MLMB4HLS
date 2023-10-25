#ifndef __DCNN_CONV2D__
#define __DCNN_CONV2D__

#include "defines.h"

namespace dcnn {

template<typename dtype, typename otype, int R, int C, int M, int K, int Pt, int PB, int Pl, int Pr, int Tm, int Tr=1, int Tc=1, Activation ACT=ReLU>
class DepthwiseConv2D {
  private:

    // typedef unsignedDataT dtype;
    typedef weightT       wtype;
    typedef biasT         btype;
    // typedef signedDataT   otype;

    typedef compactDataT<dtype, Tm> Data_t_i;
    typedef compactDataT<dtype, Tm> Data_t_o;

    typedef ac_channel<Data_t_i> chanI; // Input feature channel datatype
    typedef ac_channel<Data_t_o> chanO; // Output feature channel datatype
    typedef ac_channel<bool>     chanL; // Load enable channel datatype
    typedef ac_channel<wtype>    chanW; // Load weights channel datatype
    typedef ac_channel<btype>    chanB; // Load bias channel datatype

    typedef ndmatrix::Mat3d<dtype, M, K, K>          windowbuffer;
    typedef ndmatrix::Mat3d<dtype, C, M/Tm, K-1> linebuffer;
    typedef ndmatrix::Mat3d<wtype, M, K, K> weightbuffer;
    typedef ndmatrix::Mat1d<btype, M>          biasbuffer;
    
    windowbuffer window;
    linebuffer   lb;
    weightbuffer weights;
    biasbuffer   bias;
    
    // Updates the values inside the window and line buffers, each time
    // a new set of inputs arrive.
    void updateBuffers(dtype &pxl, const int c, const int n, const int tn) {
      _vert: for (int k = 0; k < K; k++) {
        dtype lb_out;
        if (k > 0) {
          lb[c][n/Tn][k-1][tn] = (k < K-1) ? lb[c][n/Tn][k][tn] : pxl;
          lb_out = (k < K-1) ? lb[c][n/Tn][k][tn] : pxl;
        }
        else
          lb_out = (k < K-1) ? lb[c][n/Tn][k][tn] : pxl;

        _hori: for (int l = 0; l < K; l++) {
          window[n+tn][k][l] = (l < K-1) ? window[n+tn][k][l+1] : lb_out;
        }
      }
    };
    
    // Feeds the K x K calculated products to an addition tree, in order
    // to calculate the partial output feature for a OFM using pixels
    // from a single single IFM
    otype madd(const int m, const int n) {
      otype sum = 0;
      MADi: for (int k = 0; k < K; k++) {
        otype inner_sum = 0;
        MADj: for (int l = 0; l < K; l++) {
          inner_sum += window[n][k][l] * weights[m][n][k][l];
        } // MADj
        sum += inner_sum;
      } // MADi
      return sum;
    };

  public:
    DepthwiseConv2D() {};
    ~DepthwiseConv2D() {};

    void run(chanL &l, chanW &w, chanB &b, chanI &din, chanO &dout) {

      bool load_enable;
      if(l.available(1)) {
        load_enable = l.read();
      }
      if (load_enable) {
        rd_W: for (int m = 0; m < M; m++) {
          for (int k = 0; k < K; k++) {
            for (int l = 0; l < K; l++) {
              if (w.available(1)) {
                weights[m][k][l] = w.read();
              }
            }
          }
        }

        rd_B: for (int m = 0; m < M; m++) {
          if (b.available(1)) {
            bias[m] = b.read();
          }
        }
      }

      ROWS: for (int r = 0; r < R; r+=Tr) {  // foreach row of the input feature maps
        COLS: for (int c = 0; c < C; c+=Tc) {  // foreach column of the input feature maps
          _TRS: for (int tr = 0; tr < Tr; tr++) { // foreach row of the tile
            _TCS: for (int tc = 0; tc < Tc; tc++) { // foreach column of the tile
              
              // Check if this is a border pixel
              bool isPad = ( (r + tr) < Pt) || ( (r + tr) > (R-1 - Pb) ) || 
                           ( (c + tc) < Pl) || ( (c + tc) > (C-1 - Pr) );

              ndmatrix::Mat1d<otype, M> obuff;
              for (int m = 0; m < M; m++) {
                obuff[m] = 0;
              } 

              MAP: for (int n = 0; n < N; n+=Tn) {  // foreach channel of the input features
                
                ndmatrix::Mat1d<dtype, Tn> fein;
                if (din.available(1) || isPad) {  // Here we read the Tn parallel inputs 

                  Data_t_i structured_din = (isPad) ? (Data_t_i)0 : din.read(); // Perform on-line ZeroPadding2D

                  #pragma unroll yes
                  for (int tn = 0; tn < Tn; tn++) {
                    fein[tn] = structured_din.itm[tn];
                  }
                
                  _TIM: for (int tn = 0; tn < Tn; tn++) { // foreach channel of the tile
                    updateBuffers(fein[tn], c, n, tn);

                    OMAP: for (int m = 0; m < M; m+=Tm) {  // foreach output feature map
                      compactDataT<dtype, Tm> structured_dout;
                      _TOM: for (int tm = 0; tm < Tm; tm++) { // foreach feature map of the tiled output maps
                        
                        obuff[m+tm] += madd(m+tm, n+tn);

                        if ( (n + tn) == (N-1) && ( (r + tr) >= (K-1) ) && ( (c + tc) >= (K-1) ) ) {
                          obuff[m+tm] += bias[m+tm];
                          switch (ACT) {
                            case ReLU: { 
                              structured_dout.itm[tm] = (obuff[m+tm] > 0) ? (dtype)obuff[m+tm] : (dtype)0;
                              break;
                            }
                            case Linear: { 
                              structured_dout.itm[tm] = obuff[m+tm];
                              break;
                            }
                            default: {
                              structured_dout.itm[tm] = (obuff[m+tm] > 0) ? (dtype)obuff[m+tm] : (dtype)0;
                              break;
                            }
                          } // end switch
                        } // endif

                      } // _TOM
                      if ( (n + tn) == (N-1) && ( (r + tr) >= (K-1) ) && ( (c + tc) >= (K-1) ) ) {
                        dout.write(structured_dout);
                      }
                    } // OMAP                
                  } // _TIM
                } // endif din.available(1)
              } // IMAP
            } // _TCS
          } // _TRS
        } // COLS
      } // ROWS
    }
};


}; // namespace dcnn

#endif