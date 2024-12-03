#include "dry.cuh"
#include "cuda_utils.h"

namespace aphrodite {
namespace sampling {

template <typename DType>
cudaError_t DryPenalizeLogits(
    DType* logits,                // [batch_size, vocab_size]
    const int* input_ids,         // [batch_size, seq_len]
    const int* sequence_breakers, // [batch_size, num_breakers]
    const int* num_breakers,      // [batch_size]
    const DType* multipliers,     // [batch_size]
    const DType* bases,           // [batch_size]
    const int* allowed_lengths,   // [batch_size]
    const int* ranges,            // [batch_size]
    const int batch_size,
    const int seq_len,
    const int vocab_size,
    cudaStream_t stream) {
    
    constexpr uint32_t BLOCK_THREADS = 1024;
    constexpr uint32_t VEC_SIZE = 1;
    
    const uint32_t smem_size = sizeof(DryTempStorage<DType>);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_THREADS);
    
    auto kernel = DryPenalizeLogitsKernel<BLOCK_THREADS, VEC_SIZE, DType>;
    
    // Pass seq_len and vocab_size by value
    void* args[] = {
        &logits, &input_ids, &sequence_breakers, &num_breakers,
        &multipliers, &bases, &allowed_lengths, &ranges,
        (void*)&seq_len, (void*)&vocab_size  // Cast to void* for scalar values
    };
    
    APHRODITE_CUDA_CALL(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
    
    APHRODITE_CUDA_CALL(cudaLaunchKernel(
        (void*)kernel,
        grid,
        block,
        args,
        smem_size,
        stream));
    
    return cudaSuccess;
}

// Explicit instantiations
template cudaError_t DryPenalizeLogits<float>(
    float*, const int*, const int*, const int*, const float*, const float*,
    const int*, const int*, const int, const int, const int, cudaStream_t);

template cudaError_t DryPenalizeLogits<half>(
    half*, const int*, const int*, const int*, const half*, const half*,
    const int*, const int*, const int, const int, const int, cudaStream_t);

} // namespace sampling
} // namespace aphrodite
