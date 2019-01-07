#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor SparseSelect_forward(
		const at::Tensor& features,
                const at::Tensor& batches,
		const at::Tensor& offsets,
		const int kernel_size) {
  if (features.type().is_cuda()) {
#ifdef WITH_CUDA
    return SparseSelect_forward_cuda(features, batches, offsets, kernel_size);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return SparseSelect_forward_cpu(features, batches, offsets, kernel_size);
}

int SparseSelect_backward(
		at::Tensor& d_features,
                const at::Tensor& batches,
		const at::Tensor& offsets,
		const int kernel_size, 
		const at::Tensor& d_outputs) {
  if (d_outputs.type().is_cuda()) {
#ifdef WITH_CUDA
    SparseSelect_backward_cuda(d_features, batches, offsets, kernel_size, d_outputs);
    return 0;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
