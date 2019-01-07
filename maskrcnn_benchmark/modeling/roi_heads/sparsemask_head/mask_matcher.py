import torch
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


# Generate the best matching
def generate_best_matching(anchors, targets, training=True):
    results  = []
    sparse_anchors =  []
    for anchors_per_image,  targets_per_image in zip(anchors,  targets):
        anchors_per_image = cat_boxlist(anchors_per_image)
        match_quality_matrix = boxlist_iou(targets_per_image, anchors_per_image)
        iou, matches_idx = match_quality_matrix.max(dim=1)
        select = iou >= 0.5
        matches_idx = matches_idx[select]

        results.append(matches_idx)
        sparse_anchors_per_image =  anchors_per_image[matches_idx]
        if training :
            sparse_anchors_per_image.add_field(
                'masks',
                targets_per_image.get_field('masks')[select])
            sparse_anchors_per_image.add_field(
                'labels',
                targets_per_image.get_field('labels')[select])

        sparse_anchors.append(sparse_anchors_per_image)
 
    return results, sparse_anchors
