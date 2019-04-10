/*
    SampleBuffer

    A list with additional properties that stores 32-bit integers..
*/

#ifndef _SAMPLEBUFFER_H_
#define _SAMPLEBUFFER_H_

#include <stdint.h>


class SampleBuffer {
  private:
      uint8_t counter;
      bool full;
      uint16_t capacity;
      uint8_t level;
      uint32_t * elements

  public:
      SampleBuffer(uint16_t capacity);
      ~SampleBuffer();

      bool is_full();
      void reset();
      uint8_t count();

      void set_level(uint8_t level);
      uint8_t get_level();

      bool add_element(uint32_t element);
      uint32_t* elements();
};

#endif // _SAMPLEBUFFER_H_
