import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _C


class _SparseSelect(Function):
    @staticmethod
    def forward(ctx, features, batches, offsets, kernel_size):
        ctx.save_for_backward(features, batches, offsets);
        ctx.kernel_size = kernel_size

        outputs = _C.sparse_select_forward(
            features, batches, offsets, kernel_size
        )
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, d_outputs):
        features, batches, offsets = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        d_outputs = d_outputs.contiguous()
        d_features = features.clone()
        _C.sparse_select_backward(
            d_features, batches, offsets, kernel_size, d_outputs
        )
        return d_features, None, None, None


sparse_select = _SparseSelect.apply


class SparseSelect(nn.Module):
    def __init__(self, kernel_size):
        super(SparseSelect, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, features,  batches, offsets):
        sparse_features = sparse_select(
            features, batches, offsets, self.kernel_size
        )
        return sparse_features

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "kernel_size=" + str(self.kernel_size)
        tmpstr += ")"
        return tmpstr
