// Cheng-Yang Fu
// cyfu@cs.unc.edu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void SparseSelectForward(const int nthreads, 
    const T* features,
    const int* batches,
    const int* offsets,
    const int batch, 
    const int depth,
    const int height,
    const int width,
    const int kernel_size,
    const int kernel_half,
    const int num,
    T*  outputs) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    
     int w_offset = i % kernel_size;
     int h_offset = (i / kernel_size) % kernel_size;
     int dim_idx = (i / (kernel_size  * kernel_size)) % depth;
     int mask_idx = i / (kernel_size * kernel_size * depth);

     int batch_idx = batches[mask_idx];
     int h_start = (offsets[mask_idx] / width) - kernel_half;
     int w_start = (offsets[mask_idx] % width) - kernel_half;

     int feature_h_idx = h_start + h_offset;
     int feature_w_idx = w_start + w_offset;
     int output_index = mask_idx * depth * kernel_size * kernel_size +
	     dim_idx * kernel_size * kernel_size +
	     h_offset * kernel_size + w_offset;


     if ((feature_h_idx <0) || (feature_h_idx >= height) ||
         (feature_w_idx <0) || (feature_w_idx >= width)) {
	outputs[output_index] = 0;
     }else{
        long feature_index = ((batch_idx * depth + dim_idx) * height +
                             feature_h_idx) * width + feature_w_idx;
        outputs[output_index] = features[feature_index];
    }
  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossForward


template <typename T>
__global__ void SparseSelectBackward(const int nthreads, 
    T* d_features,
    const int* batches,
    const int* offsets,
    const int batch, 
    const int depth,
    const int height,
    const int width,
    const int kernel_size,
    const int kernel_half,
    const int num,
    const T*  d_outputs) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    
     int w_offset = i % kernel_size;
     int h_offset = (i / kernel_size) % kernel_size;
     int dim_idx = (i / (kernel_size  * kernel_size)) % depth;
     int mask_idx = i / (kernel_size * kernel_size * depth);

     int batch_idx = batches[mask_idx];
     int h_start = (offsets[mask_idx] / width) - kernel_half;
     int w_start = (offsets[mask_idx] % width) - kernel_half;

     int feature_h_idx = h_start + h_offset;
     int feature_w_idx = w_start + w_offset;
     int output_index = mask_idx * depth * kernel_size * kernel_size +
	     dim_idx * kernel_size * kernel_size +
	     h_offset * kernel_size + w_offset;


     if ((feature_h_idx <0) || (feature_h_idx >= height) ||
         (feature_w_idx <0) || (feature_w_idx >= width)) {
	//do nothing
     }else{
        long feature_index = ((batch_idx * depth + dim_idx) * height +
                             feature_h_idx) * width + feature_w_idx;
	atomicAdd(d_features + feature_index, d_outputs[output_index]);
    }
  } // CUDA_1D_KERNEL_LOOP
} // SigmoidFocalLossForward


at::Tensor SparseSelect_forward_cuda(
		const at::Tensor& features,
                const at::Tensor& batches,
		const at::Tensor& offsets,
		const int kernel_size) { 
  AT_ASSERTM(features.type().is_cuda(), "features must be a CUDA tensor");
  AT_ASSERTM(batches.type().is_cuda(), "batches must be a CUDA tensor");

  const int batch = features.size(0);
  const int depth = features.size(1);
  const int height = features.size(2);
  const int width = features.size(3);
  const int num_samples = batches.size(0);
  const int kernel_half = (kernel_size - 1) / 2;
	
  auto output = at::empty({num_samples, depth, kernel_size, kernel_size}, features.options());
  auto output_size = num_samples * features.size(1) * kernel_size * kernel_size;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(features.type(), "SparseSelect_forward", [&] {
    SparseSelectForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         features.contiguous().data<scalar_t>(),
	 batches.contiguous().data<int>(),
	 offsets.contiguous().data<int>(),
         batch, depth, height, width, kernel_size, kernel_half,
	 num_samples,
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;   
}	


void SparseSelect_backward_cuda(
		at::Tensor& d_features,
                const at::Tensor& batches,
		const at::Tensor& offsets,
		const int kernel_size, 
		const at::Tensor& d_outputs) {

  AT_ASSERTM(d_outputs.type().is_cuda(), "d_outputs must be a CUDA tensor");
  AT_ASSERTM(d_features.type().is_cuda(), "d_features must be a CUDA tensor");
  AT_ASSERTM(batches.type().is_cuda(), "batches must be a CUDA tensor");

  const int batch = d_features.size(0);
  const int depth = d_features.size(1);
  const int height = d_features.size(2);
  const int width = d_features.size(3);
  const int num_samples = batches.size(0);
  const int kernel_half = (kernel_size - 1) / 2;
 
  d_features.zero_();
  auto output_size = num_samples * d_features.size(1) * kernel_size * kernel_size;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES(d_features.type(), "SparseSelect_backward", [&] {
    SparseSelectForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         d_features.contiguous().data<scalar_t>(),
	 batches.contiguous().data<int>(),
	 offsets.contiguous().data<int>(),
         batch, depth, height, width, kernel_size, kernel_half,
	 num_samples,
         d_outputs.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
}


