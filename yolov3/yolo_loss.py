from yolov3.configuration import CONFIG
import torch

def iou_wh(box_a, box_b):
	inter = torch.min(box_a, box_b) * torch.min(box_a, box_b)
	return (inter / (box_a + box_b)) - inter

def build_targets(outputs, anchors, targets):
	b, a, g, _, _ = outputs.shape # batch, anchors, grid
	mask = torch.BoolTensor(b, a, g, g, device=CONFIG.device).fill_(False)
	tx = torch.zeros(b, a, g, g, device=CONFIG.device)
	ty = torch.zeros(b, a, g, g, device=CONFIG.device)
	tw = torch.zeros(b, a, g, g, device=CONFIG.device)
	th = torch.zeros(b, a, g, g, device=CONFIG.device)
	for batch_idx in range(targets.size(0)):
		box = targets[batch_idx][...,:4] * g
		g_xy = box[:,:2]
		g_wh = box[:,2:4] / anchors[:,None]
		ious = torch.stack([iou_wh(x, g_wh) for x in anchors])
		values, indices = ious.max(0)
		mask[batch_idx, indices, g_xy[1].long(), g_xy[0].long()] = True
		tx[batch_idx, indices, g_xy[1].long(), g_xy[0].long()] = g_xy[0] - g_xy[0].floor()
		ty[batch_idx, indices, g_xy[1].long(), g_xy[0].long()] = g_xy[1] - g_xy[1].floor()
		tw[batch_idx, indices, g_xy[1].long(), g_xy[0].long()] = torch.log(g_wh[0] / (1e-8 + anchors[indices][:,0]))
		th[batch_idx, indices, g_xy[1].long(), g_xy[0].long()] = torch.log(g_wh[1] / (1e-8 + anchors[indices][:,1]))
	return mask, tx, ty, tw, th, None
