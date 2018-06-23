//-----------------------------------------------------------------------------

#ifndef _STFT_H_
#define _STFT_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include "string.h"

#include <math.h>

typedef float (*window_fct_t)(float, float);

float triangular_window(float n, float N);

float hann_window(float n, float N);

fft_complex *stft(const float *data, size_t data_size, window_fct_t window_function, size_t window_size,
                  size_t hop_size);

float *istft(fft_complex *data, size_t data_size, window_fct_t window_function, size_t window_size,
             size_t hop_size);

#ifdef __cplusplus
}
#endif

#endif

//-----------------------------------------------------------------------------