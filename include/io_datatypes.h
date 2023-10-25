/* A class that allows the use of either std::vector
*  or ac_channel in cases of C++ or HLS implementations.
*
* Created By Dionisis Filippas 17/5/2023 */
#ifndef __IO_INTERFACES__
#define __IO_INTERFACES__
#include <vector>
#include "ac_channel.h"

namespace io_interfaces {

enum INTERFACE {VECTOR, CHANNEL};

template<typename T, INTERFACE F>
class IO_DATATYPE {
private:
public:
  IO_DATATYPE() {};
  ~IO_DATATYPE() {};

}; // (end) class

template<typename T>
class IO_DATATYPE<T, VECTOR> {
private:

  typedef std::vector<T> vec_t;

  vec_t vec;
  int counter;

public:
  IO_DATATYPE() :counter(0) {};
  ~IO_DATATYPE() {};

  void write(T inp) {
    vec.push_back(inp);
    counter++;
  };

  T read() {
    T out = vec.front();
    vec.erase(vec.begin());
    counter--;
  
    return out;
  }

  bool available(int n) {
    return (counter > n-1);
  };

}; // (end) class

template<typename T>
class IO_DATATYPE<T, CHANNEL> {
private:

  ac_channel<T> chan;

public:
  IO_DATATYPE() {};
  ~IO_DATATYPE() {};

  void write(T inp) {
    chan.write(inp);
  };

  T read() {
    T out = chan.read();
    return out;
  }

  bool available(int n) {
    return chan.available(n);
  };

}; // (end) class

} // (end) namespace

#endif
