
namespace Generators {
namespace cuda {

void LaunchPopulateIndices(int* indices, int size, int batch_size, cudaStream_t stream);
void GetTopKSubset(cudaStream_t stream, float* scores_in, int* indices_out, int vocab_size, int batch_size, int k);
void GetSample(cudaStream_t stream, int32_t* d_next_token, float* d_scores, int vocab_size, int batch_size, int k, float p, float temperature);

}  // namespace cuda
}  // namespace Generators