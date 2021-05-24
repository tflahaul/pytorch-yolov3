from yolov3.configuration import CONFIG
import torch

def iou_wh(box1, box2):
	inter = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
	union = (1e-8 + (box1[0] * box1[1])) + (box2[0] * box2[1]) - inter
	return inter / union

def build_targets(outputs, anchors, targets):
	b, a, g, _, _ = outputs.shape # batch, anchors, grid
	mask = torch.BoolTensor(b, a, g, g, device=CONFIG.device).fill_(False)
	tx = torch.zeros(b, a, g, g, device=CONFIG.device)
	ty = torch.zeros(b, a, g, g, device=CONFIG.device)
	tw = torch.zeros(b, a, g, g, device=CONFIG.device)
	th = torch.zeros(b, a, g, g, device=CONFIG.device)
	tc = torch.zeros(b, a, g, g, CONFIG.classes, device=CONFIG.device)
	for batch_idx in range(targets.size(0)):
		for target in targets[batch_idx]:
			g_xy = target[:2] * g
			g_wh = target[2:4] * g
			ious = torch.stack([iou_wh(x, g_wh) for x in anchors])
			_, index = ious.max(0)
			mask[batch_idx, index, int(g_xy[0]), int(g_xy[1])] = True
			tx[batch_idx, index, int(g_xy[0]), int(g_xy[1])] = g_xy[0] - int(g_xy[0])
			ty[batch_idx, index, int(g_xy[0]), int(g_xy[1])] = g_xy[1] - int(g_xy[1])
#			tw[batch_idx, index, int(g_xy[0]), int(g_xy[1])] = torch.log(1e-8 + (g_wh[1] / anchors[index][1]))
#			th[batch_idx, index, int(g_xy[0]), int(g_xy[1])] = torch.log(1e-8 + (g_wh[0] / anchors[index][0]))
			tc[batch_idx, index, int(g_xy[0]), int(g_xy[1]), int(target[-1])] = 1.0
	return mask, tx, ty, tw, th, tc
