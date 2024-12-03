#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include "dry.cuh"

// Helper function to check CUDA errors
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error " << err << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Test configuration struct
struct TestConfig {
    int batch_size;
    int seq_len;
    int vocab_size;
    int num_breakers;
    float multiplier;
    float base;
    int allowed_len;
    int range;
    bool use_half;  // Added flag for half precision
};

template<typename T>
void copy_to_device(T* d_ptr, const std::vector<float>& h_vec) {
    if constexpr (std::is_same_v<T, float>) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_vec.data(), h_vec.size() * sizeof(float), 
                            cudaMemcpyHostToDevice));
    } else {
        std::vector<half> h_half(h_vec.size());
        for (size_t i = 0; i < h_vec.size(); i++) {
            h_half[i] = __float2half(h_vec[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, h_half.data(), h_half.size() * sizeof(half), 
                            cudaMemcpyHostToDevice));
    }
}

template<typename T>
void run_test_typed(const TestConfig& config) {
    using clock = std::chrono::high_resolution_clock;

    // Generate test data
    std::vector<int> input_ids(config.batch_size * config.seq_len);
    std::vector<int> sequence_breakers(config.batch_size * config.num_breakers);
    std::vector<int> num_breakers(config.batch_size);
    
    // Generate input sequences with repetitive patterns
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vocab_dist(0, config.vocab_size - 1);
    
    for (int i = 0; i < config.batch_size; i++) {
        for (int j = 0; j < config.seq_len; j++) {
            if (j % 3 == 0) {
                input_ids[i * config.seq_len + j] = j % 10;  // Repetitive pattern
            } else {
                input_ids[i * config.seq_len + j] = vocab_dist(gen);
            }
        }
        for (int j = 0; j < config.num_breakers; j++) {
            sequence_breakers[i * config.num_breakers + j] = vocab_dist(gen);
        }
        num_breakers[i] = config.num_breakers;
    }

    // Prepare other parameters
    std::vector<float> multipliers(config.batch_size, config.multiplier);
    std::vector<float> bases(config.batch_size, config.base);
    std::vector<int> allowed_lengths(config.batch_size, config.allowed_len);
    std::vector<int> ranges(config.batch_size, config.range);
    std::vector<float> logits(config.batch_size * config.vocab_size, 0.0f);

    // Allocate device memory
    T *d_logits;
    int *d_input_ids, *d_sequence_breakers, *d_num_breakers;
    T *d_multipliers, *d_bases;
    int *d_allowed_lengths, *d_ranges;

    CUDA_CHECK(cudaMalloc(&d_logits, logits.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_input_ids, input_ids.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sequence_breakers, sequence_breakers.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_breakers, num_breakers.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_multipliers, multipliers.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_bases, bases.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_allowed_lengths, allowed_lengths.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ranges, ranges.size() * sizeof(int)));

    // Copy data to device with proper type conversion
    copy_to_device(d_logits, logits);
    CUDA_CHECK(cudaMemcpy(d_input_ids, input_ids.data(), 
                         input_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sequence_breakers, sequence_breakers.data(), 
                         sequence_breakers.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_breakers, num_breakers.data(), 
                         num_breakers.size() * sizeof(int), cudaMemcpyHostToDevice));
    copy_to_device(d_multipliers, multipliers);
    copy_to_device(d_bases, bases);
    CUDA_CHECK(cudaMemcpy(d_allowed_lengths, allowed_lengths.data(), 
                         allowed_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranges, ranges.data(), 
                         ranges.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 5; i++) {
        aphrodite::sampling::DryPenalizeLogits(
            d_logits, d_input_ids, d_sequence_breakers, d_num_breakers,
            d_multipliers, d_bases, d_allowed_lengths, d_ranges,
            config.batch_size, config.seq_len, config.vocab_size, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iterations = 100;
    auto start = clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        aphrodite::sampling::DryPenalizeLogits(
            d_logits, d_input_ids, d_sequence_breakers, d_num_breakers,
            d_multipliers, d_bases, d_allowed_lengths, d_ranges,
            config.batch_size, config.seq_len, config.vocab_size, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float ms_per_iter = duration / 1000.0f / num_iterations;
    
    std::cout << "Test configuration:" << std::endl
              << "  Batch size: " << config.batch_size << std::endl
              << "  Sequence length: " << config.seq_len << std::endl
              << "  Vocab size: " << config.vocab_size << std::endl
              << "  Precision: " << (std::is_same_v<T, half> ? "half" : "float") << std::endl
              << "Performance:" << std::endl
              << "  Average time per iteration: " << ms_per_iter << " ms" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_input_ids));
    CUDA_CHECK(cudaFree(d_sequence_breakers));
    CUDA_CHECK(cudaFree(d_num_breakers));
    CUDA_CHECK(cudaFree(d_multipliers));
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_allowed_lengths));
    CUDA_CHECK(cudaFree(d_ranges));
}

int main() {
    // Test different configurations
    std::vector<TestConfig> configs = {
        // Small batch, float
        {1, 512, 32000, 4, 1.0f, 1.1f, 4, 0, false},
        // Small batch, half
        {1, 512, 32000, 4, 1.0f, 1.1f, 4, 0, true},
        // Medium batch, float
        {4, 512, 32000, 4, 1.0f, 1.1f, 4, 0, false},
        // Medium batch, half
        {4, 512, 32000, 4, 1.0f, 1.1f, 4, 0, true},
        // Large batch, float
        {16, 512, 32000, 4, 1.0f, 1.1f, 4, 0, false},
        // Large batch, half
        {16, 512, 32000, 4, 1.0f, 1.1f, 4, 0, true},
        // Long sequence, float
        {4, 2048, 32000, 4, 1.0f, 1.1f, 4, 0, false},
        // Long sequence, half
        {4, 2048, 32000, 4, 1.0f, 1.1f, 4, 0, true},
    };

    for (const auto& config : configs) {
        if (config.use_half) {
            run_test_typed<half>(config);
        } else {
            run_test_typed<float>(config);
        }
        std::cout << std::endl;
    }

    return 0;
}