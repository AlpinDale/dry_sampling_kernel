#ifndef APHRODITE_DRY_CUH_
#define APHRODITE_DRY_CUH_

#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include "vec_dtypes.cuh"
#include "utils.cuh"

namespace aphrodite {
namespace sampling {

__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
            __float_as_uint(max(val, __uint_as_float(assumed))));
    } while (assumed != old);
    return __uint_as_float(old);
}

// Custom atomicMax for half
#if __CUDA_ARCH__ >= 700
__device__ __forceinline__ half atomicMaxHalf(half* address, half val) {
    unsigned short int* address_as_uint = (unsigned short int*)address;
    unsigned short int old = *address_as_uint;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
            __half_as_ushort(__float2half(
                max(__half2float(val), 
                    __half2float(__ushort_as_half(assumed))))));
    } while (assumed != old);
    return __ushort_as_half(old);
}
#endif

template <typename DType>
__device__ __forceinline__ DType atomic_max(DType* address, DType val);

template <>
__device__ __forceinline__ float atomic_max(float* address, float val) {
    return atomicMaxFloat(address, val);
}

template <>
__device__ __forceinline__ half atomic_max(half* address, half val) {
#if __CUDA_ARCH__ >= 700
    return atomicMaxHalf(address, val);
#else
    return __float2half(0.0f);  // Fallback for older architectures
#endif
}

template <typename DType>
__device__ __forceinline__ DType type_zero() {
    if constexpr (std::is_same_v<DType, float>) {
        return 0.0f;
    } else {
        return __float2half(0.0f);
    }
}

template <typename DType>
__device__ __forceinline__ DType pow_custom(DType base, int exp) {
    if constexpr (std::is_same_v<DType, float>) {
        return powf(base, static_cast<float>(exp));
    } else {
        return __float2half(powf(__half2float(base), static_cast<float>(exp)));
    }
}

template <typename DType>
struct DryTempStorage {
    union {
        DType z_array[1024];
        DType penalties[1024];
    };
    int max_match_length;
};

// Compute Z-array in parallel for pattern matching
template <uint32_t BLOCK_THREADS>
__device__ void computeZArray(
    const int* input_ids,
    const int start_idx,
    const int length,
    int* z_array,
    const int* sequence_breakers,
    const int num_breakers) {
    
    int left = 0, right = 0;
    const int tid = threadIdx.x;
    
    for (int i = tid; i < length; i += BLOCK_THREADS) {
        if (i == 0) {
            z_array[i] = length;
            continue;
        }
        
        bool is_breaker = false;
        for (int j = 0; j < num_breakers; ++j) {
            if (input_ids[start_idx + i] == sequence_breakers[j]) {
                is_breaker = true;
                break;
            }
        }
        if (is_breaker) {
            z_array[i] = 0;
            continue;
        }

        if (i > right) {
            left = right = i;
            while (right < length && 
                   input_ids[start_idx + right] == input_ids[start_idx + right - i]) {
                right++;
            }
            z_array[i] = right - left;
            right--;
        } else {
            int k = i - left;
            if (z_array[k] < right - i + 1) {
                z_array[i] = z_array[k];
            } else {
                left = i;
                while (right < length && 
                       input_ids[start_idx + right] == input_ids[start_idx + right - i]) {
                    right++;
                }
                z_array[i] = right - left;
                right--;
            }
        }
    }
    __syncthreads();
}

template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType>
__global__ void DryPenalizeLogitsKernel(
    DType* logits,
    const int* input_ids,
    const int* sequence_breakers,
    const int* num_breakers,
    const DType* multipliers,
    const DType* bases,
    const int* allowed_lengths,
    const int* ranges,
    const int seq_len,
    const int vocab_size) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const DType multiplier = multipliers[batch_idx];
    if (__heq(multiplier, type_zero<DType>())) return;  // For half
    
    const DType base = bases[batch_idx];
    const int allowed_len = allowed_lengths[batch_idx];
    const int range = ranges[batch_idx];
    const int seq_breakers_count = num_breakers[batch_idx];
    
    const int search_len = (range == 0) ? seq_len : min(seq_len, range);
    
    extern __shared__ uint8_t smem[];
    DryTempStorage<DType>* storage = reinterpret_cast<DryTempStorage<DType>*>(smem);
    int* z_array = reinterpret_cast<int*>(storage->z_array);
    
    computeZArray<BLOCK_THREADS>(
        input_ids + batch_idx * seq_len,
        seq_len - search_len,
        search_len,
        z_array,
        sequence_breakers + batch_idx * seq_breakers_count,
        seq_breakers_count);
    
    if (tid < vocab_size) {
        storage->penalties[tid] = type_zero<DType>();
    }
    __syncthreads();
    
    for (int i = tid; i < search_len - 1; i += BLOCK_THREADS) {
        if (z_array[i] >= allowed_len) {
            const int next_token = input_ids[batch_idx * seq_len + i + 1];
            if (next_token < vocab_size) {
                const DType penalty = multiplier * 
                    pow_custom(base, z_array[i] - allowed_len);
                atomic_max(storage->penalties + next_token, penalty);
            }
        }
    }
    __syncthreads();
    
    const int items_per_thread = (vocab_size + BLOCK_THREADS - 1) / BLOCK_THREADS;
    const int start_idx = tid * items_per_thread;
    const int end_idx = min(start_idx + items_per_thread, vocab_size);
    
    for (int i = start_idx; i < end_idx; i++) {
        const DType penalty = storage->penalties[i];
        // Use type-safe comparison
        if (__hgt(penalty, type_zero<DType>())) {
            logits[batch_idx * vocab_size + i] -= penalty;
        }
    }
}

} // namespace sampling
} // namespace aphrodite

#endif // APHRODITE_DRY_CUH_
