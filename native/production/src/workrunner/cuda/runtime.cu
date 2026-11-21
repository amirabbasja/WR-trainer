#include <cuda_runtime.h>
extern "C" const char* wr_cuda_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
