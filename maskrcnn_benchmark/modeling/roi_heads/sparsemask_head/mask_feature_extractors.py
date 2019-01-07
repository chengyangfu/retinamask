# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import SparseSelect

class SparseMaskFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(SparseMaskFPNFeatureExtractor, self).__init__()

        #resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        #scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        #sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        #pooler = Pooler(
        #    output_size=(resolution, resolution),
        #    scales=scales,
        #    sampling_ratio=sampling_ratio,
        #)
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        #self.pooler = pooler
        layers = cfg.MODEL.SPARSE_MASK_HEAD.CONV_LAYERS

        next_feature = input_size
        self.blocks = []
        self.gn = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "sparsemask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

            if cfg.MODEL.USE_GN:
                layer_name = "sparsemask_gn{}".format(layer_idx)
                module = nn.GroupNorm(32, layer_features)
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                self.add_module(layer_name, module)
                self.gn.append(layer_name)

    def forward(self, features, batch_idx, layer_idx,  locations):

        # Before Sparse Selection
        index = torch.arange(0, len(batch_idx), dtype=torch.long).to(batch_idx.device)
        post_index = []
        post_features = []
        for layer,  feature in enumerate(features):
            x = feature
            for idx, layer_name in enumerate(self.blocks):
                x = getattr(self, layer_name)(x)
                if self.gn:
                    x = getattr(self, self.gn[idx])(x)

                x = F.relu(x)

            select = layer_idx == layer
            if select.sum().item():
                select_features = SparseSelect(3)(x, batch_idx[select].int(),
                                                  locations[select].int())
                post_features.append(select_features)
                post_index.append(index[select])

        post_index = torch.cat(post_index)
        post_features = torch.cat(post_features, 0)
        post_features = post_features[post_index]
        return post_features


_SPARSE_MASK_FEATURE_EXTRACTORS = {
    "SparseMaskFPNFeatureExtractor": SparseMaskFPNFeatureExtractor,
}


def make_sparse_mask_feature_extractor(cfg):
    func = _SPARSE_MASK_FEATURE_EXTRACTORS[cfg.MODEL.SPARSE_MASK_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
