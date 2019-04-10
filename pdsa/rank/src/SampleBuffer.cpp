#include "SampleBuffer.h"

SampleBuffer(uint16_t capacity) {
    elements = malloc(capacity * sizeof(uint32_t));
    SampleBuffer::reset();
}

~SampleBuffer(uint16_t capacity) {
    if (elements != NULL) {
        dealloc(elements)
    }
}

void SampleBuffer::reset() {
    counter = 0;
    full = 0;
    level   = 0;
}

bool SampleBuffer::is_full() {
    return full;
}

uint8_t SampleBuffer::count() {
    return counter;
}

uint8_t SampleBuffer::set_level(uint8_t new_level) {
    level = new_level;
}

uint8_t SampleBuffer::get_level() {
    return level;
}

bool SampleBuffer::add_element(uint32_t element) {
    if(counter >= capacity) {
        return false;
    }
    elements.add(element);
    full = true;
    return true;
}

uint32_t* SampleBuffer::elements() {
    return elements + counter * sizeof(uint32_t);
}
