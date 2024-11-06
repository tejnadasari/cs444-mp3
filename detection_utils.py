import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox):
    """
    Args:
        anchor: batch of anchors in the format (x1, y1, x2, y2) or in other words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes image batch size and A denotes the number of anchors
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
    Hint: remember if the max_iou for that bounding box is between [0, 0.4) then the gt_cls should equal 0(because it is being assigned as background) and the
    gt_bbox should be all zero(it can be anything since it will be ignored however our tests set them to zero so you should too).
    Also, if the max iou is between [0.4, 0.5) then the gt_cls should be equal to -1(since it's neither background or assigned to a class. This is basically tells the model to ignore this box) 
    and the gt_bbox should again arbitrarilarly be set to all zeros).
    Otherwise if the max_iou > 0.5, you should assign the anchor to the gt_box with the max iou, and the gt_cls will be the ground truth class of that max_iou box
    Hint: use torch.max to get both the max iou and the index of the max iou.

    Hint: We recommend using the compute_bbox_iou function which efficently computes the ious between two lists of bounding boxes as a helper function.

    Hint: make sure that the returned gt_clss tensor is of type int(since it will be used as an index in the loss function). Also make sure that both the gt_bboxes and gt_clss are on the same device as the anchor. 
    You can do this by calling .to(anchor.device) on the tensor you want to move to the same device as the anchor.

    VECTORIZING CODE: Again, you can use for loops initially to make the tests pass, but in order to make your code efficient 
    during training, you should only have one for loop over the batch dimension and everything else should be vectorized. We recommend using boolean masks to do this. i.e
    you can compute the max_ious for all the anchor boxes and then do gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set to background and setting their gt_cls to 0. 
    This will remove the need for a for loop over all the anchor boxes. You can then do the same for the other cases. This will make your code much more efficient and faster to train.
    """
    # TODO(student): Complete this function
    # TEJNA
    B = anchor.shape[0]  # batch size
    A = anchor.shape[1]  # number of anchors

    # Initialize targets with same device as anchor and correct shape (B, A, 1)
    gt_clss = torch.zeros((B, A, 1), dtype=torch.float, device=anchor.device)
    gt_bboxes = torch.zeros((B, A, 4), device=anchor.device)

    for b in range(B):
        # Skip if no objects in this image
        if cls[b].numel() == 0:
            continue

        # Compute IoU between all anchors and all ground truth boxes
        ious = compute_bbox_iou(anchor[b], bbox[b])  # Shape: (num_anchors, num_gt_boxes)

        # Get maximum IoU and corresponding GT box index for each anchor
        max_ious, max_idx = torch.max(ious, dim=1)  # Shape: (num_anchors,)

        # Assign targets based on IoU thresholds
        # Background: IoU < 0.4
        background_mask = max_ious < 0.4
        gt_clss[b][background_mask, 0] = 0

        # Ignore: 0.4 <= IoU < 0.5
        ignore_mask = (max_ious >= 0.4) & (max_ious < 0.5)
        gt_clss[b][ignore_mask, 0] = -1

        # Positive: IoU >= 0.5
        positive_mask = max_ious >= 0.5
        gt_clss[b][positive_mask, 0] = cls[b][max_idx[positive_mask]].float().squeeze(-1)

        # Assign bbox targets only for positive anchors
        if positive_mask.any():
            matched_gt_boxes = bbox[b][max_idx[positive_mask]]
            gt_bboxes[b][positive_mask] = matched_gt_boxes

    return gt_clss.to(torch.int), gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors of shape (A, 4)
        gt_bboxes: groundtruth object classes of shape (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    
    Remember that the delta_x and delta_y we compute are with respect to the center of the anchor box. I.E, we're seeing how much that center of the anchor box changes. 
    We also need to normalize delta_x and delta_y which means that we need to divide them by the width or height of the anchor box respectively. This is to make
    our regression targets more invariant to the size of the original anchor box. So, this means that:
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be computed in a similar manner.

    When computing delta_w and delta_h, there are a few things to note.
    1. We also want to normalize these with respect to the width and height of the anchor boxes. so delta_w = gt_bbox_width / anchor_width
    2. Logarithm: In order to make our regresssion targets better handle varying sizees of the bounding boxes, we use the logarithmic scale for our delta_w and delta_h
       This is to ensure that if for example the gt_width is twice or 1/2 the size of the anchor_width, the magnitude in the log scale would stay the same but only the sign of
       our regression target would be different. Therefore our formula changes to delta_w = log(gt_bbox_width / anchor_width)
    3. Clamping: Remember that logarithms can't handle negative values and that the log of values very close to zero will have very large magnitudes and have extremly 
       high gradients which might make training unstable. To mitigate this we use clamping to ensure that the value that we log isn't too small. Therefore, our final formula will be
       delta_w = log(max(gt_bbox_width,1) / anchor_width)
       
    """
    # TODO(student): Complete this function - TEJNA

    # Convert boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height)
    # Anchors
    anchor_widths = anchors[..., 2] - anchors[..., 0]
    anchor_heights = anchors[..., 3] - anchors[..., 1]
    anchor_ctr_x = anchors[..., 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[..., 1] + 0.5 * anchor_heights

    # Ground truth boxes
    gt_widths = gt_bboxes[..., 2] - gt_bboxes[..., 0]
    gt_heights = gt_bboxes[..., 3] - gt_bboxes[..., 1]
    gt_ctr_x = gt_bboxes[..., 0] + 0.5 * gt_widths
    gt_ctr_y = gt_bboxes[..., 1] + 0.5 * gt_heights

    # Compute deltas
    # Center coordinates
    delta_x = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    delta_y = (gt_ctr_y - anchor_ctr_y) / anchor_heights

    # Width and height
    # First clamp the gt widths/heights to minimum of 1
    gt_widths = torch.clamp(gt_widths, min=1.0)
    gt_heights = torch.clamp(gt_heights, min=1.0)

    # Then compute log ratios
    delta_w = torch.log(gt_widths / anchor_widths)
    delta_h = torch.log(gt_heights / anchor_heights)

    return torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
    """
    # Extract widths and heights of anchor boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # Extract predicted deltas
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    # Apply deltas to center coordinates
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y

    # Apply deltas to width and height
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # Convert back to (x1, y1, x2, y2) format
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    # Stack to get final boxes
    new_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

    return new_boxes

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    
    Remember that nms is used to prevent having many boxes that overlap each other. To do this, if multiple boxes overlap each other beyond a
    threshold iou, nms will pick the "best" box(the one with the highest score) and remove the rest. One way to implement this is to
    first compute the ious between all pairs of bboxes. Then loop over the bboxes from highest score to lowest score. Since this is the 
    best bbox(the one with the highest score), It will be choosen over all overlapping boxes. Therefore, you should add this bbox to your final 
    resulting bboxes and remove all the boxes that overlap with it from consideration. Then repeat until you've gone through all of the bboxes.

    make sure that the indices tensor that you return is of type int or long(since it will be used as an index to select the relevant bboxes to output)
    """
    # TODO(student): Complete this function - TEJNA
    # If no boxes, return empty tensor
    if bboxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=bboxes.device)

    # Get indices that would sort scores in descending order
    _, order = scores.sort(0, descending=True)

    # Initialize list to keep track of indices to keep
    keep = []

    while order.numel() > 0:
        # Pick the box with highest score
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i.item())

        # Get IoUs of all remaining boxes with the currently selected box
        curr_box = bboxes[i].unsqueeze(0)  # (1, 4)
        other_boxes = bboxes[order[1:]]  # (n-1, 4)
        ious = compute_bbox_iou(curr_box, other_boxes).squeeze(0)  # (n-1,)

        # Keep only boxes with IoU less than threshold
        mask = ious <= threshold
        order = order[1:][mask]

    # Convert to tensor
    keep = torch.tensor(keep, dtype=torch.int64, device=bboxes.device)

    return keep
