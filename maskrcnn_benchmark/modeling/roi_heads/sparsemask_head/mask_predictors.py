# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


class SparseMaskFCPredictor(nn.Module):
    def __init__(self, cfg):
        super(SparseMaskFCPredictor, self).__init__()
        self.num_anchors = 9
        num_inputs = 256

        self.mask_fc_logits = []
        for i in range(self.num_anchors):
            module = Conv2d(num_inputs, 28*28, 3, 1, 0)
            layer_name = "sparsemask_pred{}".format(i)
            self.add_module(layer_name, module)
            self.mask_fc_logits.append(layer_name)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, anchor_idx):
        _, depth, height, width= x.shape
        index = torch.arange(0, len(anchor_idx),
                            dtype=torch.long).to(x[0].device)
        results = []
        post_index = []
        for a, layer_name in enumerate(self.mask_fc_logits):
            select = a == anchor_idx
            if select.sum().item():
                results.append(
                    getattr(self, layer_name)(x[select].view(-1, depth, height, width)))
                post_index.append(index[select])

        post_index = torch.cat(post_index)
        results = torch.cat(results, 0)
        results = results[post_index]
        return results


_SPARSE_MASK_PREDICTOR = {"SparseMaskFCPredictor": SparseMaskFCPredictor}

def make_sparse_mask_predictor(cfg):
    #func = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    func = _SPARSE_MASK_PREDICTOR["SparseMaskFCPredictor"]
    return func(cfg)
