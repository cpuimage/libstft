
#include "fft.h"
#include "stft.h"

float triangular_window(float n, float N) {
    return 1.0f - fabsf((n - (N - 1) * 0.5f) / (N * 0.5f));
}

float hann_window(float n, float N) {
    const float pi_weight = 2.0f * 3.14159265358979323846f;
    return (0.5f - 0.5f * cosf(pi_weight * n / (N - 1)));
}

fft_complex *stft(const float *data, size_t data_size, window_fct_t window_function, size_t window_size,
                  size_t hop_size) {
    size_t result_size = (data_size / hop_size);
    float *_data = (float *) calloc(window_size, sizeof(float));
    fft_complex *_stft = (fft_complex *) calloc(window_size, sizeof(fft_complex));
    fft_complex * result = (fft_complex *) calloc(result_size * window_size, sizeof(fft_complex));
    if (_data == NULL || _stft == NULL || result == NULL) {
        if (_data) free(_data);
        if (_stft) free(_stft);
        if (result) free(result);
        return NULL;
    }
    fft_plan forward = fft_plan_dft_r2c_1d(window_size, _data, _stft, FFT_ESTIMATE);
    if (forward.input != NULL && forward.ip != NULL && forward.w != NULL) {
        size_t idx = 0;
        for (size_t pos = 0; pos < data_size; pos += hop_size) {
            for (size_t i = 0; i < window_size; ++i) {
                if (pos + i < data_size)
                    _data[i] = window_function(i, window_size) * data[pos + i];
                else
                    _data[i] = 0;
            }
            fft_execute(forward);
            memcpy(result + idx * window_size, _stft, sizeof(fft_complex) * window_size);
            idx++;
        }
    }
    fft_destroy_plan(forward);
    free(_data);
    free(_stft);
    return result;
}

float *istft(fft_complex *data, size_t data_size, window_fct_t window_function, size_t window_size,
             size_t hop_size) {
    size_t result_size = data_size * hop_size + (window_size - hop_size);
    float *frame = (float *) calloc(window_size, sizeof(float));
    fft_complex *slice = (fft_complex *) malloc(sizeof(fft_complex) * window_size);
    float *result = (float *) calloc(result_size, sizeof(float));
    if (frame == NULL || slice == NULL || result == NULL) {
        if (frame) free(frame);
        if (slice) free(slice);
        if (result) free(result);
        return NULL;
    }
    fft_plan istft = fft_plan_dft_c2r_1d(window_size, slice, frame, FFT_ESTIMATE);
    if (istft.input != NULL && istft.ip != NULL && istft.w != NULL) {
        for (size_t i = 0; i < data_size; ++i) {
            memcpy(slice, data + i * window_size, sizeof(fft_complex) * window_size);
            fft_execute(istft);
            for (size_t pos = 0; pos < window_size; ++pos) {
                size_t r_pos = pos + i * hop_size;
                result[r_pos] += frame[pos] * window_function(pos, window_size);
            }
        }
    }
    fft_destroy_plan(istft);
    free(frame);
    free(slice);
    return result;
}