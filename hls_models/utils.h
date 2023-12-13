#ifndef __UTILS__
#define __UTILS__


/* *** Addition binary tree *** */
template<int N>
struct addition {
  template<typename T>
  T add(T A[N]) {
    T B[N/2 + N%2];

    #pragma hls_unroll
    for (int i = 0; i <N/2; i++) {
      B[i] = A[2*i] + A[2*i+1];
    }

    if (N%2 == 1)
      B[N/2] = A[N-1];
      
    addition<N/2 + N%2> adder;
    
    return adder.add(B);
  }
};

template<>
struct addition<2> {
  template<typename T>
  T add(T A[2]) {
    T B = A[0] + A[1];
    return B;
  }
};

template<>
struct addition<1> {
  template<typename T>
  static T add(T *a) {
    T res = a[0];
    return res;
  }
};

template<int N, typename T>
T add_tree(T A[N]) {
  addition<N> adder;
  return adder.add(A);
};


/* *** Maximum binary tree *** */
template<int N>
struct max_r {
  template<typename T>
  static T max(T *a) {
    T m0 = max_s<N/2>::max(a);
    T m1 = max_s<N-N/2>::max(a+N/2);

    return m0 > m1 ? m0 : m1;
  }
};

template<> 
struct max_r<1> {
  template<typename T>
  static T max(T *a) {
    return a[0];
  }
};

template<int N, typename T>
T max_tree(T *a) {
  return max_r<N>::max(a);
};

#endif
