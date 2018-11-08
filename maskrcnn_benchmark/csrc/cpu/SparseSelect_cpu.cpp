// Cheng-Yang Fu
// cyfu@cs.unc.edu
#include "cpu/vision.h"

template <typename T>
void SparseSelectForward_cpu_kernel(
    const int nthreads, 
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
    
    int n_masks = nthreads / (kernel_size * kernel_size * depth);


     for (int mask_idx=0; mask_idx < n_masks; mask_idx++) {
	for (int dim_idx =0; dim_idx < depth; dim_idx++) {
           for (int h_offset=0; h_offset < kernel_size; h_offset++) {
              for (int w_offset=0; w_offset < kernel_size; w_offset++) {

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

	      }
	   }
	}
     }
} // SigmoidFocalLossForward


at::Tensor SparseSelect_forward_cpu(
		const at::Tensor& features,
                const at::Tensor& batches,
		const at::Tensor& offsets,
		const int kernel_size) { 
  AT_ASSERTM(!features.type().is_cuda(), "features must be a CPU tensor");
  AT_ASSERTM(!batches.type().is_cuda(), "batches must be a CPU tensor");

  const int batch = features.size(0);
  const int depth = features.size(1);
  const int height = features.size(2);
  const int width = features.size(3);
  const int num_samples = batches.size(0);
  const int kernel_half = (kernel_size - 1) / 2;
	
  auto output = at::empty({num_samples, depth, kernel_size, kernel_size}, features.options());
  auto output_size = num_samples * features.size(1) * kernel_size * kernel_size;

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(features.type(), "SparseSelect_forward", [&] {
    SparseSelectForward_cpu_kernel<scalar_t>(
         output_size,
         features.contiguous().data<scalar_t>(),
	 batches.contiguous().data<int>(),
	 offsets.contiguous().data<int>(),
         batch, depth, height, width, kernel_size, kernel_half,
	 num_samples,
         output.data<scalar_t>());
  });
  return output;   
}	


