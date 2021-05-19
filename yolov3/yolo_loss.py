from yolov3.configuration import CONFIG

import torch

def iou(box_a, box_b):
	ax1, ay1, ax2, ay2 = box_a.t()
	bx1, by1, bx2, by2 = box_b.t()
	x1 = torch.max(ax1, bx1)
	x2 = torch.min(ax2, bx2)
	y1 = torch.max(ay1, by1)
	y2 = torch.min(ay2, by2)
	inter = (1 + x2 - x1) * (1 + y2 - y1)
	a_area = (1 + ax2 - ax1) * (1 + ay2 - ay1)
	b_area = (1 + bx2 - bx1) * (1 + by2 - by1)
	return inter / (1e-8 + (a_area + b_area - inter))

def nms(outputs):
	results = list([None] * len(outputs))
	for index, out in enumerate(outputs):
		out = out[out[:,:,4] > CONFIG.confidence_thres]
		if out.size(0) > 0:
			score = out[:,4] * out[:,4:].max(1)[0]
			out = out[(-score).argsort()]
			cls_conf, cls_pred = out[:,4:].max(1, keepdim=True)
			detections = torch.cat((out[:,:4], cls_conf, cls_pred), 1)
			keep_boxes = list()
			while detections.size(0) > 0:
				ious = iou(detections[0,:4].unsqueeze(0), detections[:,:4])
				invalid_mask = (ious > CONFIG.nms_thres) & (detections[0,-1] == detections[:,-1])
				w = detections[invalid_mask,4:5]
				detections[0,:4] = (w * detections[invalid_mask,:4]).sum(0) / w.sum()
				keep_boxes += detections[0]
			if keep_boxes:
				results[index] = torch.stack(keep_boxes)
	return results

def build_targets(outputs, targets):
	pass
