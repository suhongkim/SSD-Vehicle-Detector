import torch
import numpy as np
''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''


def generate_prior_bboxes(prior_layer_cfg):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

    Use VGG_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. Conv4    | (38x38)        | (30x30) (unit. pixels)
    2. Conv7    | (19x19)        | (60x60)
    3. Conv8_2  | (10x10)        | (111x111)
    4. Conv9_2  | (5x5)          | (162x162)
    5. Conv10_2 | (3x3)          | (213x213)
    6. Conv11_2 | (1x1)          | (264x264)
    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
    """
    example_prior_layer_cfg = [
        # Example:
        {'layer_name': 'Conv4', 'feature_dim_hw': (64, 64), 'bbox_size': (60, 60), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
        {'layer_name': 'Conv4', 'feature_dim_hw': (64, 64), 'bbox_size': (60, 60), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
        # ...
        # TODO: define your feature map settings
    ]

    priors_bboxes = []
    for feat_level_idx in range(0, len(prior_layer_cfg)):               # iterate each layers
        layer_cfg = prior_layer_cfg[feat_level_idx]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']

        # Todo: compute S_{k} (reference: SSD Paper equation 4.)
        s_min = 0.1
        s_max = 0.9
        k = feat_level_idx + 1
        m = len(prior_layer_cfg)

        sk = s_min + ((s_max - s_min) / (m - 1)) * (k - 1)
        fk = layer_feature_dim[0]

        for y in range(0, layer_feature_dim[0]):
            for x in range(0,layer_feature_dim[0]):
                # Todo: compute bounding box center
                cx = (x + 0.5) / fk
                cy = (y + 0.5) / fk

                # Todo: generate prior bounding box with respect to the aspect ratio
                for aspect_ratio in layer_aspect_ratio:
                    h = sk / np.sqrt(aspect_ratio)
                    w = sk * np.sqrt(aspect_ratio)
                    priors_bboxes.append([cx, cy, w, h])

    # Convert to Tensor
    priors_bboxes = torch.tensor(priors_bboxes)
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)
    num_priors = priors_bboxes.shape[0]

    if torch.cuda.is_available():
        priors_bboxes = priors_bboxes.cuda()

    # [DEBUG] check the output shape
    assert priors_bboxes.dim() == 2
    assert priors_bboxes.shape[1] == 4
    return priors_bboxes


def iou(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4)
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """
    # [DEBUG] Check if input is the desire shape
    assert a.dim() == 2
    assert a.shape[1] == 4
    assert b.dim() == 2
    assert b.shape[1] == 4

    if torch.cuda.is_available():
        a = a.cuda()
        b = b.cuda()

    # TODO: implement IoU of two bounding box
    # change inputs to corner-form
    a = torch.clamp(center2corner(a), min=0, max=1)  # input is normalized(0~1)
    b = torch.clamp(center2corner(b), min=0, max=1)  # input is normalized(0~1)

    left_top = torch.max(a[:, :2], b[:, :2])
    right_bottom = torch.min(a[:, 2:], b[:, 2:])
    inter_w = torch.clamp(right_bottom[:, 0]-left_top[:, 0], min=0, max=1)  # negative value -> 0
    inter_h = torch.clamp(right_bottom[:, 1]-left_top[:, 1], min=0, max=1)  # negative value -> 0
    inter_area = inter_w * inter_h

    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    over_area = a_area + b_area - inter_area

    iou = torch.div(inter_area, over_area)

    # [DEBUG] Check if output is the desire shape
    assert iou.dim() == 1
    assert iou.shape[0] == a.shape[0]
    return iou


def match_priors(prior_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor, iou_threshold: float):
    """
    Match the ground-truth boxes with the priors.
    Note: Use this function in your ''cityscape_dataset.py', see the SSD paper page 5 for reference. (note that default box = prior boxes)

    :param gt_bboxes: ground-truth bounding boxes, dim:(n_samples, 4)
    :param gt_labels: ground-truth classification labels, negative (background) = 0, dim: (n_samples)
    :param prior_bboxes: prior bounding boxes on different levels, dim:(num_priors, 4)
    :param iou_threshold: matching criterion
    :return matched_boxes: real matched bounding box, dim: (num_priors, 4)
    :return matched_labels: real matched classification label, dim: (num_priors)
    """
    # [DEBUG] Check if input is the desire shape
    assert gt_bboxes.dim() == 2
    assert gt_bboxes.shape[1] == 4
    assert gt_labels.dim() == 1
    assert gt_labels.shape[0] == gt_bboxes.shape[0]
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    if torch.cuda.is_available():
        prior_bboxes = prior_bboxes.cuda()
        gt_bboxes = gt_bboxes.cuda()
        gt_labels = gt_labels.cuda()

    # TODO: implement prior matching
    ious = torch.zeros(prior_bboxes.shape[0], gt_bboxes.shape[0])
    for obj_idx, gt_bbox in zip(range(gt_bboxes.shape[1]), gt_bboxes):
        ious[:, obj_idx] = iou(prior_bboxes, gt_bbox.unsqueeze(0)) #prior shape: 4 -> 1x4 (unsqueeze)
    # which prior has max iou per each target: (n_objects, )
    max_iou_object, max_iou_object_idx = ious.max(0)
    # which target has max iou per each prior (n_priors, )
    max_iou_prior, max_iou_prior_idx = ious.max(1)

    # In case of all iou is less then threshold,
    #   need to reserve the index of max iou
    #   so that each object has at least one matched default box
    #   (because of the calculation of loss function (matching) )
    for obj_index, prior_index in enumerate(max_iou_object_idx):
        max_iou_prior_idx[prior_index] = obj_index
    max_iou_prior[max_iou_object_idx] = 1  # boost iou value to make sure that matched boxes >= n_objects

    # generate (n_priors, ) label array with value of gt_labels
    matched_labels = gt_labels[max_iou_prior_idx].clone()  # iou matrix-> (n_prior, 1) with max iou object value
    # discard labels under threshold (background:0)
    matched_labels[max_iou_prior < iou_threshold] = 0  # Background

    matched_boxes = gt_bboxes[max_iou_prior_idx].clone()  # matched gt_bbox

    # [DEBUG] Check if output is the desire shape
    assert matched_boxes.dim() == 2
    assert matched_boxes.shape[1] == 4
    assert matched_labels.dim() == 1
    assert matched_labels.shape[0] == matched_boxes.shape[0]

    return matched_boxes, matched_labels


''' NMS ----------------------------------------------------------------------------------------------------------------
'''
def nms_bbox(bbox_loc, bbox_confid_scores, overlap_threshold=0.4, prob_threshold=0.255, top_k = -1, candidate_size=200):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param bbox_loc: bounding box loc and size, dim: (num_priors, 4)
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes)
    :param overlap_threshold: the overlap threshold for filtering out outliers
    :return: selected bounding box with classes
    """

    # [DEBUG] Check if input is the desire shape
    assert bbox_loc.dim() == 2
    assert bbox_loc.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_loc.shape[0]

    sel_bbox = []

    # Todo: implement nms for filtering out the unnecessary bounding boxes
    num_classes = bbox_confid_scores.shape[1]
    bboxes = torch.clamp(center2corner(bbox_loc), min=0, max=1)

    for class_idx in range(1, num_classes):
        # choose candidate
        scores = bbox_confid_scores[:, class_idx]
        sorted_score, indexes = scores.sort(0, descending=True)
        indexes = indexes[sorted_score > prob_threshold]
        picked = []
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            current_box = bboxes[current, :]
            indexes = indexes[1:]
            rest_boxes = bboxes[indexes, :]
            iou_output = iou(
                rest_boxes,
                current_box.unsqueeze(0),
            )
            indexes = indexes[iou_output <= overlap_threshold]

        sel_bbox.append({'class': class_idx,
                         'picked_scores': scores[picked],
                         'picked_bboxes': bboxes[picked]})

    return sel_bbox


''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''


def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    """
    Compute SSD predicted locations to boxes(cx, cy, h, w).
    :param loc: predicted location, dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: boxes: (cx, cy, h, w)
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    l_center = loc[..., :2]
    l_size = loc[..., 2:]

    # real bounding box
    return torch.cat([
        center_var * l_center * p_size + p_center,      # b_{center}
        p_size * torch.exp(size_var * l_size)           # b_{size}
    ], dim=-1)


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    """
    Compute boxes (cx, cy, h, w) to SSD locations form.
    :param bbox: bounding box (cx, cy, h, w) , dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: loc: (cx, cy, h, w)
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]

    return torch.cat([
        1 / center_var * ((b_center - p_center) / p_size),
        torch.log(b_size / p_size) / size_var
    ], dim=-1)


def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([center[..., :2] - center[..., 2:]/2,
                      center[..., :2] + center[..., 2:]/2], dim=-1)


def corner2center(corner):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([corner[..., :2] - corner[..., 2:]/2,
                      corner[..., :2] + corner[..., 2:]/2], dim=-1)