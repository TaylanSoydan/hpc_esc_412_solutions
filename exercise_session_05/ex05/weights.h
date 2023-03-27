#include <cmath>

inline int ngp_weights(float x, float *W) {
    int i = std::floor(x);
    W[0] = 1.0;
    return i;
}

inline int cic_weights(float x, float *W) {
    int i = std::floor(x-0.5);
    float i_start = 1.0 * i + 0.5;
    float s0 = x - i_start; 
    W[0] = 1 - s0; // fix me
    W[1] = s0; // fix me
    return i;
}

inline int tsc_weights(float x, float *W) {
    int i = std::floor(x-1.0);
    float i_start = 1.0 * i + 0.5;
    float s0 = x - i_start;
    float s1 = std::abs(x - i_start - 1);
    float s2 = i_start + 2 - x;
    W[0] = 0.5 * (1.5 - s0) * (1.5 - s0); // fix me
    W[1] = 0.75 - s1 * s1; // fix me
    W[2] = 0.5 * (1.5 - s2) * (1.5 - s2); // fix me
    return i;
}

inline int pcs_weights(float x, float *W) {
    int i = std::floor(x-1.5);
    float i_start = 1.0 * i + 0.5;
    float s0 = x - i_start;
    float s1 = x - i_start - 1;
    float s2 = i_start + 2 - x;
    float s3 = i_start + 3 - x;

    W[0] = 1.0/6.0 * (2-s0) * (2-s0) * (2-s0); // fix me
    W[1] = 1.0/6.0 * (4 - 6 * s1 * s1 + 3 * s1 * s1 * s1); // fix me
    W[2] = 1.0/6.0 * (4 - 6 * s2 * s2 + 3 * s2 * s2 * s2); // fix me
    W[3] = 1.0/6.0 * (2-s3) * (2-s3) * (2-s3);
    return i;
}

