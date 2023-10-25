#ifndef __DCNN_HELPERS__
#define __DCNN_HELPERS__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "io_datatypes.h"
// #include <ac_channel.h>

template<typename T, size_t N, size_t M>
void print2Darray(T** X) {
  for (size_t i=0; i<N; i++) {
    std::cout << "Line ";
    if (N > 10 && i<10)
      std::cout << " ";
    std::cout << i << ":\t";
    for (size_t j=0; j<M; j++) {
      std::cout << X[i][j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
};

template<typename T, size_t N>
void print1Darray(T* X) {
  for (size_t i=0; i<N; i++) 
    std::cout << X[i] << " ";
  std::cout << std::endl;
};

// Used for Bias read
template<class T, size_t SIZE>
void read_1d_array_from_txt(T* w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    size_t i = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        while(std::getline(iss, token, ',')) {
            float x;
            std::istringstream(token) >> x;
            // if (SIZE == 1 && i<1) 
                w[i] = x;
            i++;
        }

    }
}

template<class T, size_t SIZE>
void read_1d_ndarray_from_txt(ndmatrix::Mat1d<T,SIZE> &w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    size_t i = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        while(std::getline(iss, token, ',')) {
            float x;
            std::istringstream(token) >> x;
            // if (SIZE == 1 && i<1) 
                w[i] = x;
            i++;
        }

    }
}

// Used for Dense Weights
template<class T, size_t SIZE_IF, size_t SIZE_OF>
void read_2d_array_from_txt(T** w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    
    size_t i = 0;
    size_t j = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        
        while(std::getline(iss, token, ',')) {
            

            float x;
            std::istringstream(token) >> x;
            w[i][j] = x;

            i = (j < SIZE_OF-1) ? i : i+1;
            j = (j < SIZE_OF-1) ? j+1 : (size_t)0;
        }


    }
}

template<class T, size_t SIZE_IF, size_t SIZE_OF>
void read_2d_ndarray_from_txt(ndmatrix::Mat2d<T,SIZE_IF,SIZE_OF> &w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    
    size_t i = 0;
    size_t j = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        
        while(std::getline(iss, token, ',')) {
            

            float x;
            std::istringstream(token) >> x;
            w[i][j] = x;

            i = (j < SIZE_OF-1) ? i : i+1;
            j = (j < SIZE_OF-1) ? j+1 : (size_t)0;
        }


    }
}

// Used for input image and Normalization scale and bias
template<class T, size_t SIZE_H, size_t SIZE_W, size_t SIZE_C>
void read_3d_array_from_txt(T*** w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }
    
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        
        while(std::getline(iss, token, ',')) {

            // std::istringstream(token) >> w[i][j][k][l];
            float x;
            std::istringstream(token) >> x;
            w[i][j][k] = x;

            k = (i == (SIZE_H-1) && j == (SIZE_W-1)) ? k+1 : k;
            i = (j < SIZE_W-1) ? i : ((i < SIZE_H-1) ? i+1 : (size_t)0);
            j = (j < SIZE_W-1) ? j+1 : (size_t)0;
        }      

    
    }
};

template<class T, size_t SIZE_H, size_t SIZE_W, size_t SIZE_C>
void read_3d_ndarray_from_txt(ndmatrix::Mat3d<T,SIZE_H,SIZE_W,SIZE_C> &w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }
    
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        
        while(std::getline(iss, token, ',')) {

            // std::istringstream(token) >> w[i][j][k][l];
            float x;
            std::istringstream(token) >> x;
            w[i][j][k] = x;

            k = (i == (SIZE_H-1) && j == (SIZE_W-1)) ? k+1 : k;
            i = (j < SIZE_W-1) ? i : ((i < SIZE_H-1) ? i+1 : (size_t)0);
            j = (j < SIZE_W-1) ? j+1 : (size_t)0;
        }      

    
    }
};

// Used for Conv2D Weights
template<class T, size_t SIZE_OF, size_t SIZE_IF, size_t SIZE_K, size_t SIZE_L>
void read_4d_array_from_txt(T**** w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    size_t l = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        
        
        while(std::getline(iss, token, ',')) {

            float x;
            std::istringstream(token) >> x;
            w[i][j][k][l] = x;
            


            i = (j == (SIZE_IF-1) && k == (SIZE_K-1) && l == (SIZE_L-1)) ? i+1 : i;
            j = (k == (SIZE_K-1) && l == (SIZE_L-1)) ? ((j < SIZE_IF-1) ? j+1 : (size_t)0) : j;
            k = (l < SIZE_L-1) ? k : ((k < SIZE_K-1) ? k+1 : (size_t)0);
            l = (l < SIZE_L-1) ? l+1 : (size_t)0;
        }

    }
}

template<class T, size_t SIZE_OF, size_t SIZE_IF, size_t SIZE_K, size_t SIZE_L>
void read_4d_ndarray_from_txt(ndmatrix::Mat4d<T,SIZE_OF,SIZE_IF,SIZE_K,SIZE_L> &w, const char* fname) {

    std::string full_path = std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    size_t l = 0;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;
        
        
        while(std::getline(iss, token, ',')) {

            float x;
            std::istringstream(token) >> x;
            w[i][j][k][l] = x;
            


            i = (j == (SIZE_IF-1) && k == (SIZE_K-1) && l == (SIZE_L-1)) ? i+1 : i;
            j = (k == (SIZE_K-1) && l == (SIZE_L-1)) ? ((j < SIZE_IF-1) ? j+1 : (size_t)0) : j;
            k = (l < SIZE_L-1) ? k : ((k < SIZE_K-1) ? k+1 : (size_t)0);
            l = (l < SIZE_L-1) ? l+1 : (size_t)0;
        }

    }
}

// Used for output
template<typename T, size_t SIZE>
void write_1d_array_to_txt(T* w, const char* fname) {

   std::ofstream outdata; // outdata is like cin

    outdata.open(std::string(fname).c_str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
    }

    for (size_t i=0; i<SIZE; ++i)
        outdata << w[i] << std::endl;

    outdata.close();

};

template<typename T, size_t SIZE>
void write_1d_ndarray_to_txt(ndmatrix::Mat1d<T,SIZE> &w, const char* fname) {

   std::ofstream outdata; // outdata is like cin

    outdata.open(std::string(fname).c_str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
    }

    for (size_t i=0; i<SIZE; ++i)
        outdata << w[i] << std::endl;

    outdata.close();

};

template<typename T, size_t SH, size_t SW, size_t SC>
void write_image_to_channel(ac_channel<T> &ic, const char* fname) {
  T*** inAr;
    inAr = new T**[SH];
    for (size_t i=0; i<SH; i++) {
      inAr[i] = new T*[SW];
      for (size_t j=0; j<SW; j++)
        inAr[i][j] = new T[SC];
    }

    read_3d_array_from_txt<T,SH, SW, SC>(inAr, fname);

    for (size_t i=0; i<SH; i++) 
      for (size_t j=0; j<SW; j++) 
        for (size_t m=0; m<SC; m++) 
          ic.write(inAr[i][j][m]); 

    for (size_t i=0; i<SH; i++) {
      for (size_t j=0; j<SW; j++)
        delete[] inAr[i][j];
      delete[] inAr[i];
    }
    delete[] inAr;
}

template<typename T, size_t OF, size_t IF, size_t K1, size_t K2>
void write_weights_to_channel(ac_channel<T> &wc, const char* fname) {
    T**** weights;
    weights = new T***[OF];
    for (size_t i=0; i<OF; i++) {
      weights[i] = new T**[IF];
      for (size_t j=0; j<IF; j++) {
        weights[i][j] = new T*[K1];
        for (size_t k=0; k<K1; k++)
          weights[i][j][k] = new T[K2];
      }
    }

    read_4d_array_from_txt<T,OF,IF,K1,K2>(weights, fname);

    for (size_t i=0; i<OF; i++) 
      for (size_t j=0; j<IF; j++) 
        for (size_t k=0; k<K1; k++) 
          for (size_t l=0; l<K2; l++) 
            wc.write(weights[i][j][k][l]);
          

    for (size_t i=0; i<OF; i++) {
      for (size_t j=0; j<IF; j++) {
        for (size_t k=0; k<K1; k++)
          delete[] weights[i][j][k];
        delete[] weights[i][j];
      }
      delete[] weights[i];
    }
    delete[] weights;
};

template<typename T, size_t IF, size_t OF>
void write_dense_weights_to_channel(ac_channel<T> &wc, const char* fname) {
    T** weights;
    weights = new T*[IF];
    for (size_t i=0; i<IF; i++) 
      weights[i] = new T[OF];

    read_2d_array_from_txt<T,IF,OF>(weights, fname);

    for (size_t i=0; i<IF; i++) 
      for (size_t j=0; j<OF; j++) 
        wc.write(weights[i][j]);
          

    for (size_t i=0; i<IF; i++) 
      delete[] weights[i];
    delete[] weights;
};

template<typename T, size_t OF>
void write_bias_to_channel(ac_channel<T> &bc, const char* fname) {
     T* bias = new T[OF];

    read_1d_array_from_txt<T,OF>(bias, fname);

    for (size_t i=0; i<OF; i++) 
      bc.write(bias[i]); 

    delete[] bias;
};


template<typename T, size_t SH, size_t SW, size_t SC>
void write_3d_array_to_txt(T*** w, const char* fname) {

   std::ofstream outdata; // outdata is like cin

    outdata.open(std::string(fname).c_str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
    }

    for (size_t k=0; k<SC; k++) {
      for (size_t i=0; i<SH; i++) {
        for (size_t j=0; j<SW; j++) {
          if (j<SW-1) {
            #ifdef __USE_FF4HLS__
            outdata << w[i][j][k].to_float() << ",";
            #else
            outdata << w[i][j][k] << ",";
            #endif
            
          }
          else {
            #ifdef __USE_FF4HLS__
            outdata << w[i][j][k].to_float() << std::endl;
            #else
            outdata << w[i][j][k] << std::endl;
            #endif
          }
        }
      }
    }
    outdata.close();
};

template<typename T, size_t SH, size_t SW, size_t SC>
void write_3d_ndarray_to_txt(ndmatrix::Mat3d<T, SH, SW, SC> &w, const char* fname) {

   std::ofstream outdata; // outdata is like cin

    outdata.open(std::string(fname).c_str()); // opens the file
    if( !outdata ) { // file couldn't be opened
        std::cerr << "Error: file could not be opened" << std::endl;
        exit(1);
    }

    for (size_t k=0; k<SC; k++) {
      for (size_t i=0; i<SH; i++) {
        for (size_t j=0; j<SW; j++) {
          if (j<SW-1) {
            #ifdef __USE_FF4HLS__
            outdata << w[i][j][k].to_float() << ",";
            #else
            outdata << w[i][j][k] << ",";
            #endif
            
          }
          else {
            #ifdef __USE_FF4HLS__
            outdata << w[i][j][k].to_float() << std::endl;
            #else
            outdata << w[i][j][k] << std::endl;
            #endif
          }
        }
      }
    }
    outdata.close();
};

template<typename T, size_t SH, size_t SW, size_t SC>
void write_3d_channel_data_to_txt(ac_channel<T> &i_chan, const char* fname) {

  T*** arr = new T**[SH];
  for (size_t i=0; i<SH; i++) {
    arr[i] = new T*[SW];
    for (size_t j=0; j<SW; j++) 
      arr[i][j] = new T[SC];
  }

  size_t i=0, j=0, k=0;
  while (i_chan.available(1)) {
    arr[i][j][k] = i_chan.read();

    k++;
    if (k==SC) {
      k=0;
      j++;
      if (j==SW) {
        j=0;
        i++;
      }
    }
  }

  write_3d_array_to_txt<T,SH,SW,SC>(arr, fname);

  for (size_t i=0; i<SH; i++) {
    for (size_t j=0; j<SW; j++) 
      delete[] arr[i][j];
    delete[] arr[i];
  }
  delete[] arr;
};


template<typename T, size_t SIZE>
void write_1d_channel_data_to_txt(ac_channel<T> &i_chan, const char* fname) {

  T* arr = new T[SIZE];

  size_t i=0;
  while (i_chan.available(1)) {
    arr[i] = i_chan.read();

    i++;
  }

  write_1d_array_to_txt<T,SIZE>(arr, fname);

  delete[] arr;
};


#endif
