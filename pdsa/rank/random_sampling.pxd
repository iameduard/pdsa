from libc.stdint cimport uint32_t, uint16_t, uint8_t

cdef extern from "src/SampleBuffer.h":
   cdef cppclass SampleBuffer:
    #   uint8_t counter
    #   bint full
    #   uint16_t capacity
    #   uint8_t level
    #   uint32_t* elements

      SampleBuffer(uint16_t capacity)


      void reset()
      uint8_t count()
      bint is_full()

      void set_level(uint8_t level)
      uint8_t get_level()

      bint add_element(uint32_t element)
      uint32_t* elements()


cdef class RandomSampling:
    cdef uint16_t number_of_buffers
    cdef uint16_t buffer_capacity
    cdef uint8_t height

    cdef SampleBuffer* _buffers

    cpdef void consume(self, object dataset)

    cpdef uint32_t quantile_query(self, float quantile) except *
    cpdef size_t inverse_quantile_query(self, uint32_t element) except *
    cpdef size_t interval_query(self, uint32_t start, uint32_t end) except *

    cpdef size_t sizeof(self)
    cpdef size_t count(self)

    cdef uint8_t _active_level(self)
    cdef void _collapse(self, uint8_t active_level)
    cdef uint16_t _get_next_empty_buffer_id(self, uint8_t active_level)
