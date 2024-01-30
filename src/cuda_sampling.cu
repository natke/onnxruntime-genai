// Filter Logits Kernel implmentation from ONNX Runtime

// #include "generators.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include "cuda_sampling.cuh"
#include "smartptrs.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>

namespace Generators {
namespace cuda {

const int max_threads = 1024;
constexpr int GPU_WARP_SIZE = 32;
constexpr int GPU_WARP_SIZE_HOST = 32;

////////////// SOFTMAX KERNELS AND LAUNCHER //////////////

// TODO: why do we need all these template struct things, can't we just do this in the main kernel???
// TODO: factor out what we can
template <typename T, typename AccumT>
struct MaxFloat {
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T, typename AccumT>
struct SumExpFloat {
  __device__ __forceinline__ SumExpFloat(AccumT v)
      : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + exp((AccumT)v - max_k);
  }

  const AccumT max_k;
};

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

// aligned vector generates vectorized load/store on CUDA
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

template <template <typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT ilpReduce(int shift, T* data, int size, const Reduction<T, 
                                            AccumT>& r, AccumT defaultVal) {
  using LoadT = aligned_vector<T, ILP>;
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;
  // shift and do 1
  if (shift > 0) {
    data -= shift;
    size += shift;
    if (threadIdx.x >= shift && threadIdx.x < size) {
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  if (size <= 0) return threadVal;
  int last = size % (ILP * blockDim.x);
  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);
  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }
  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);
  return threadVal;
}

template <template <typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT blockReduce(AccumT* smem, AccumT val, const Reduction<AccumT>& r, AccumT defaultVal) {
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();
  smem[threadIdx.x] = val;
  __syncthreads();
  AccumT warpVal = defaultVal;
  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < GPU_WARP_SIZE) {
    int warps_per_block = blockDim.x / GPU_WARP_SIZE;
    for (int i = 0; i < warps_per_block; ++i) {
      warpVal = r(warpVal, smem[i * GPU_WARP_SIZE + threadIdx.x]);
    }
    smem[threadIdx.x] = warpVal;
  }
  __syncthreads();
  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int i = 0; i < GPU_WARP_SIZE; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }
  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

dim3 SoftMax_getBlockSize(int ILP, uint64_t size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = min(size / ILP, static_cast<uint64_t>(max_threads));
  // In the vectorized case we want to trade off allowing more of the buffers to be accessed
  // in a vectorized way against wanting a larger block size to get better utilisation.
  // In general with ILP you can have (ILP-1)/ILP of the buffer accessed vectorised, at the risk
  // of having a very small block size. We choose to keep >= 1/2 of the buffer vectorised while
  // allowing a larger block size.
  if (ILP > 1) {
    max_block_size /= 2;
  }
  while (block_size < max_block_size) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = max(block_size, static_cast<uint64_t>(GPU_WARP_SIZE_HOST));
  return dim3(static_cast<unsigned int>(block_size));
}

// TODO: see if we can simplify this whole softmax situation
template <typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), logsum(log(sum)) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>((AccumT)input - max_input - logsum);
  }

  const AccumT max_input;
  const AccumT logsum;
};

template <typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(exp((AccumT)input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
};

/**
 * This will apply the Epilogue with vectorized reads & writes when input & output have the same shift
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__device__ __forceinline__ void WriteFpropResultsVectorized(int size,
                                                            const int shift,
                                                            scalar_t* input,
                                                            outscalar_t* output,
                                                            Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  using LoadT = aligned_vector<scalar_t, ILP>;
  using StoreT = aligned_vector<outscalar_t, ILP>;
  int offset = threadIdx.x;
  // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
  if (shift > 0) {
    input -= shift;
    output -= shift;
    size += shift;
    if (threadIdx.x >= shift && threadIdx.x < size) {
      output[offset] = epilogue(input[offset]);
    }
    size -= blockDim.x;
    input += blockDim.x;
    output += blockDim.x;
  }
  if (size <= 0) return;
  const int last = size % (ILP * blockDim.x);
  scalar_t in_v[ILP];
  LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);
  outscalar_t out_v[ILP];
  StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);
  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *in_value = reinterpret_cast<LoadT*>(input)[offset];
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      out_v[j] = epilogue(in_v[j]);
    }
    reinterpret_cast<StoreT*>(output)[offset] = *out_value;
  }
  offset = size - last + threadIdx.x;
  // handle the tail
  for (; offset < size; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

/**
 * This will apply the Epilogue with non-vectrorized reads & writes for the general case
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__device__ __forceinline__ void WriteFpropResults(int classes,
                                                  scalar_t* input,
                                                  outscalar_t* output,
                                                  Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  // Main bulk of loop with ILP
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
    }
  }
  // Remainder - no ILP
  for (; offset < classes; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t,
          template <typename, typename, typename> class Epilogue>
__global__ void softmax_block_forward(outscalar_t* output, scalar_t* input, int classes,
                                      int input_stride, int output_stride, accscalar_t temperature) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;
  const int input_align_bytes = ILP * sizeof(scalar_t);
  const int output_align_bytes = ILP * sizeof(outscalar_t);
  const int shift = ((uint64_t)input) % input_align_bytes / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % output_align_bytes / sizeof(outscalar_t);
  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), -std::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = blockReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -std::numeric_limits<accscalar_t>::max());
  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k/temperature), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));
  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);
  if (shift == output_shift) {
    WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
  } else {
    WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
  }
}

template <bool is_log_softmax>
void dispatch_blockwise_softmax_forward(cudaStream_t* stream, float* output, const float* input, int softmax_elements,
                                          int input_stride, int output_stride, int batch_count, float temperature=1.0) {
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(float);
  dim3 block = SoftMax_getBlockSize(ILP, softmax_elements);
  if (is_log_softmax) {
    softmax_block_forward<ILP, float, float, float, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), *stream>>>(output, const_cast<float*>(input),
                                                           softmax_elements, input_stride, output_stride, temperature);
  } else {
    softmax_block_forward<ILP, float, float, float, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), *stream>>>(output, const_cast<float*>(input),
                                                           softmax_elements, input_stride, output_stride, temperature);
  }
}

/////////////// POPULATE KERNEL LAUNCHERS ////////////////

__global__ void populate_indices(int* indices, int size) {
  int index = threadIdx.x;
  for (; index < size; index += blockDim.x) {
    int offset = index + blockIdx.x * size;
    indices[offset] = index;
  }
}

void LaunchPopulateIndices(int* indices, int size, int batch_size, cudaStream_t stream) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(256, 1, 1);
  populate_indices<<<grid, block, 0, stream>>>(indices, size);
}

__global__ void populate_offsets(int* offsets, int size) {
  int index = threadIdx.x;
  offsets[index] = index * size;
}

// TODO: large version
void launch_populate_offsets(int* offsets, int size, int batch_size, cudaStream_t stream) {
  dim3 grid(1, 1, 1);
  dim3 block(batch_size+1, 1, 1);
  populate_offsets<<<grid, block, 0, stream>>>(offsets, size);
}

/////////////// SORTING KERNEL LAUNCHER //////////////////

template <typename T>
void LaunchSortPairs(void* d_temp_storage,
                     size_t temp_storage_bytes,
                     const T* d_keys_in,
                     T* d_keys_out,
                     const int* d_values_in,
                     int* d_values_out,
                     int num_items,
                     int num_segments,
                     int* d_offsets,
                     cudaStream_t stream,
                     bool is_descending) {
  if (is_descending) {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
        d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
  } else {
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
        d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
  }
}

template <typename T>
void GetTempStorageSize(const T* d_keys_in,
                        const int* d_values_in,
                        int* d_offsets,
                        int num_items,
                        int num_segments,
                        cudaStream_t stream,
                        bool is_descending,
                        size_t& temp_storage_bytes) {
  if (is_descending) {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, (T*)nullptr,
        d_values_in, (int*)nullptr, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
  } else {
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys_in, (T*)nullptr,
        d_values_in, (int*)nullptr, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
}
}

/////////////// SAMPLE KERNEL LAUNCHER ///////////////////

template <int kBlockSize>
__global__ void prefix_sum_kernel(float* scores, float* prefix_sums, int sample_range, int vocab_size, int batch_size) {
  int batch = blockIdx.x;
  float prefix_sum = 0.0f;

  typedef cub::BlockScan<float, kBlockSize> BlockScan;
__shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = 0; i < sample_range; i += blockDim.x) {
    int global_index = threadIdx.x + i + batch * vocab_size;
    int local_index = threadIdx.x + i;
    float score = (local_index < sample_range) ? scores[global_index] : 0.0f;
    float sum = score;
    BlockScan(temp_storage).InclusiveSum(sum, sum);
    prefix_sum += sum;
    if (local_index < sample_range) {
      prefix_sums[local_index + batch * sample_range] = prefix_sum;
    }
    __syncthreads();
  }
}

__global__ void get_topk(int* indices_in, int* indices_out, int batch_size, int vocab_size, int k) {
  int index_out = threadIdx.x + blockIdx.x * blockDim.x;
  int index_local = index_out % k;
  int batch = index_out / k;

  if (batch < batch_size) {
    indices_out[index_out] = indices_in[index_local + batch * vocab_size];
  }
}

// Sets up random thresholds for top p or top k sampling
__global__ void random_threshold_kernel(int seed, float* thresholds, float* prefix_sums, int batch_size, float p, int k) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  curandState state;
  curand_init(seed, index, 0, &state);

  float k_prob = k == -1 ? 1.0 : prefix_sums[k-1];
  if (index < batch_size) {
    float min_p = fminf(p, k_prob);
    thresholds[index] = min_p * curand_uniform(&state);
  }
}

// TODO: is constexpr a good idea here for some of the if statements?
// TODO: Also, right now it's top p with k as a fallback, but we could also do top k sampled with p thresholds
template <int kBlockSize>
__global__ void simple_sample_kernel(float* prefix_sums, int* indices, int* index_out, int sample_range, int vocab_size, float* thresholds) {
  int batch = blockIdx.x;
  int index = threadIdx.x;

  __shared__ int first_index;
  if (threadIdx.x == 0) {
    first_index = sample_range - 1;
  }
  __syncthreads();

  for (; index < sample_range; index += blockDim.x) {
    float sum = prefix_sums[index + batch * sample_range];
    if (thresholds != nullptr) {
      // TOP P or K
      if (sum >= thresholds[batch]) {
        atomicMin(&first_index, index);
      } else if (index == sample_range - 1) {
        atomicMin(&first_index, index);
      }
    } else {
      // TOP 1
      first_index = 0;
      break;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    index_out[batch] = indices[first_index + batch * vocab_size];
  }
}

void LaunchSampleKernel(cudaStream_t stream, float* scores, int* indices, int* index_out, int vocab_size, int batch_size, int p = 0.0, int k=-1) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(256, 1, 1);
  int sample_range = k < 1 || k > vocab_size ? vocab_size : k;
  // Prefix Sums
  auto prefix_sums_buffer = CudaMallocHostArray<float>(sample_range * batch_size);
  std::span<float> prefix_sums{prefix_sums_buffer.get(), static_cast<size_t>(sample_range * batch_size)};
  prefix_sum_kernel<256><<<grid, block, 0, stream>>>(scores, prefix_sums.data(), sample_range, vocab_size, batch_size);
  std::span<float> thresholds;
  // Random Thresholds for Top P or Top K Sampling
  if (k > 1 || p > 0.0) {
    auto thresholds_buffer = CudaMallocHostArray<float>(batch_size);
    thresholds = std::span<float>(thresholds_buffer.get(), static_cast<size_t>(batch_size));
    std::random_device rd;
    std::mt19937 eee(rd());
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
    int seed = dist(eee);
    random_threshold_kernel<<<int(batch_size / 256) + 1, 256, 0, stream>>>(seed, thresholds.data(), prefix_sums.data(), batch_size, p > 0.0 ? p : 1.0, k > 1 ? k : -1);

    // std::vector<float> thresholds_cpu(batch_size);
    // cudaMemcpy(thresholds_cpu.data(), thresholds.data(), batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < batch_size; i++) {
    //   printf("Thresholds: %f\n", thresholds_cpu[i]);
    // }
  }
  // Do Sampling
  simple_sample_kernel<256><<<grid, block, 0, stream>>>(prefix_sums.data(), indices, index_out, sample_range, vocab_size, thresholds.data());
  
  // std::vector<int> index_out_cpu(batch_size);
  // cudaMemcpy(index_out_cpu.data(), index_out, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < batch_size; i++) {
  //   printf("Sampled Index: %d\n", index_out_cpu[i]);
  // }
}

// TODO: cuda code error checking
/////////////// TOP P+K KERNEL LAUNCHER ////////////////////

// Outputs sorted scores and corresponding indices... scores_out and indices_out should already be allocated
void SoftmaxAndSort(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, float temperature) {
  // Softmax scores
  auto scores_buffer = CudaMallocHostArray<float>(vocab_size * batch_size);
  std::span<float> scores{scores_buffer.get(), static_cast<size_t>(vocab_size * batch_size)};
  dispatch_blockwise_softmax_forward<false>(&stream, scores.data(), const_cast<const float*>(scores_in), vocab_size, vocab_size, vocab_size, batch_size, temperature);
  // Sort indices by scores
  auto offsets_buffer = CudaMallocHostArray<int>(batch_size + 1);
  std::span<int> offsets_gpu{offsets_buffer.get(), static_cast<size_t>(batch_size + 1)};
  launch_populate_offsets(offsets_gpu.data(), vocab_size, batch_size, stream);
  auto indices_buffer = CudaMallocHostArray<int>(vocab_size * batch_size);
  std::span<int32_t> indices_in{indices_buffer.get(), static_cast<size_t>(vocab_size * batch_size)};
  LaunchPopulateIndices(indices_in.data(), vocab_size, batch_size, stream);
  size_t temp_storage_bytes = 0;
  GetTempStorageSize<float>(scores.data(), indices_in.data(), offsets_gpu.data(), vocab_size * batch_size,
                            batch_size, stream, /*is_descending*/true, temp_storage_bytes);
  auto temp_buffer = CudaMallocHostArray<float>(temp_storage_bytes / sizeof(float));
  std::span<float> temp_span{temp_buffer.get(), temp_storage_bytes / sizeof(float)};
  LaunchSortPairs<float>(temp_span.data(), temp_storage_bytes, scores.data(), scores_out,
                         indices_in.data(), indices_out, vocab_size * batch_size, batch_size, offsets_gpu.data(),
                         stream, /*is_descending*/true);
}

void GetTopKSubset(cudaStream_t stream, float* scores_in, int* indices_out, int vocab_size, int batch_size, int k) {
  // Softmax and sort scores
  auto scores_buffer = CudaMallocHostArray<float>(vocab_size * batch_size);
  auto indices_buffer = CudaMallocHostArray<int>(vocab_size * batch_size);
  std::span<float> scores{scores_buffer.get(), static_cast<size_t>(vocab_size * batch_size)};
  std::span<int> indices_in{indices_buffer.get(), static_cast<size_t>(vocab_size * batch_size)};
  SoftmaxAndSort(stream, scores_in, scores.data(), indices_in.data(), vocab_size, batch_size, 1.0);
  // Get top k subset
  dim3 grid(batch_size * int(1 + k / 256), 1, 1);
  dim3 block(256, 1, 1);
  get_topk<<<grid, block, 0, stream>>>(indices_in.data(), indices_out, batch_size, vocab_size, k);
}

// Kernel launcher for combined (or seperate) top k and top p sampling; where k is the max number of tokens to sample and p is the probability threshold
void GetSample(cudaStream_t stream, int32_t* next_token_out, float* scores_in, int vocab_size, int batch_size, int k, float p, float temperature) {
  // Softmax and sort scores
  auto scores_buffer = CudaMallocHostArray<float>(vocab_size * batch_size);
  auto indices_buffer = CudaMallocHostArray<int>(vocab_size * batch_size);
  std::span<float> scores{scores_buffer.get(), static_cast<size_t>(vocab_size * batch_size)};
  std::span<int> indices{indices_buffer.get(), static_cast<size_t>(vocab_size * batch_size)};
  SoftmaxAndSort(stream, scores_in, scores.data(), indices.data(), vocab_size, batch_size, temperature);
  // Sample kernel
  LaunchSampleKernel(stream, scores.data(), indices.data(), next_token_out, vocab_size, batch_size, p, k);
}

} // namespace cuda
} // namespace Generators