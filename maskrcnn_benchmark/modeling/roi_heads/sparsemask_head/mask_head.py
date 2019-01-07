import torch
from torch import nn
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .mask_feature_extractors import make_sparse_mask_feature_extractor
from .mask_predictors import make_sparse_mask_predictor
#from .inference import make_roi_mask_post_processor
from .loss import make_sparse_mask_loss_evaluator
from .mask_matcher import generate_best_matching


class SparseMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(SparseMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_sparse_mask_feature_extractor(cfg)
        self.num_anchors = len(cfg.RETINANET.ASPECT_RATIOS) \
                            * cfg.RETINANET.SCALES_PER_OCTAVE
        self.predictor = make_sparse_mask_predictor(cfg)
        #self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_sparse_mask_loss_evaluator(cfg)
        self.discretize = 28

    def change_to_anchor_masks(self, anchors, scale=1.5):
        for i in range(len(anchors)):
            temp = anchors[i].bbox
            hw = temp[:, 2:] - temp[:, 0:2] 
            xy  = (temp[:, 2:] + temp[:, :2]) * 0.5
            hw  *= scale
            temp[:, 0:2] = xy - hw*0.5
            temp[:, 2:] = xy + hw*0.5
            anchors[i].bbox = temp
        return anchors

    def forward(self, features, anchors, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            anchors (list[BoxList]): anchor boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        device = features[0].device
        if self.training:
            with torch.no_grad():
                sparse_codes, sparse_anchors = generate_best_matching(
                    anchors, targets, self.training)
                sparse_batch = [torch.LongTensor(s.size()).fill_(idx) for idx, s in enumerate(sparse_codes)]
                sparse_batch = torch.cat(sparse_batch).to(device)
                sparse_codes = torch.cat(sparse_codes)
                layers  = torch.cat(
                    [torch.empty(f.size(2)*f.size(3),
                                 dtype=torch.long).fill_(l) for
                                     l, f in enumerate(features)])
                sparse_layers = layers[sparse_codes / self.num_anchors]
                layer_base  = torch.LongTensor(
                    [0] + [f.size(2)*f.size(3) for f in
                           features]).cumsum(0).to(device)
                sparse_off = (sparse_codes / self.num_anchors) - layer_base[sparse_layers]
                sparse_anchor_idx = sparse_codes % self.num_anchors
        else:
            sparse_batch = torch.cat([a.get_field('sparse_batch') for a \
                                     in anchors])
            sparse_layers = torch.cat([a.get_field('sparse_layers') for a \
                                     in anchors])
            sparse_off = torch.cat([a.get_field('sparse_off') for a \
                                     in anchors])
            sparse_anchor_idx = torch.cat(
                [ a.get_field('sparse_anchor_idx') for a in anchors])

            sparse_anchors = [BoxList(a.get_field('sparse_anchors'), a.size, mode="xyxy") for a in anchors]

        x = self.feature_extractor(
            features, sparse_batch,  sparse_layers, sparse_off
        )

        logits = self.predictor(x, sparse_anchor_idx)
        logits = logits.view(-1, self.discretize, self.discretize)
        sparse_anchors = self.change_to_anchor_masks(sparse_anchors)


        if not self.training:
            return features, logits, {}

        loss_mask = self.loss_evaluator(logits, sparse_anchors)
        return features, sparse_anchors, dict(loss_mask=loss_mask)

def build_sparse_mask_head(cfg):
    return SparseMaskHead(cfg)
