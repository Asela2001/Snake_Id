import numpy as np

def postprocess(outputs, conf_thres=0.25, iou_thres=0.45):
    """
    outputs: numpy array with shape (batch, num_preds, 6)
             where each pred: [x, y, w, h, conf, class_id]
    """
    # If batch dimension exists, remove it (assuming batch=1)
    if len(outputs.shape) == 3:
        outputs = outputs[0]

    boxes = outputs[:, :4]  # x, y, w, h
    scores = outputs[:, 4]  # confidence scores
    class_ids = outputs[:, 5].astype(int)

    # Filter by confidence threshold
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return []

    # Convert xywh to xyxy (top-left and bottom-right corners)
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Apply Non-Maximum Suppression (NMS)
    indices = nms(boxes_xyxy, scores, iou_thres)

    # Prepare final detections
    detections = []
    for i in indices:
        det = np.concatenate((boxes_xyxy[i], [scores[i], class_ids[i]]))
        detections.append(det)

    return detections


def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS)
    boxes: [N,4] in xyxy format
    scores: [N]
    Returns indices of kept boxes
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
