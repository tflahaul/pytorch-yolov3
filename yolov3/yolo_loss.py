from yolov3.configuration import CONFIG
import torch

class YOLOv3Loss(torch.nn.Module):
	def __init__(self, ignore_thres=0.5) -> None:
		super(YOLOv3Loss, self).__init__()
		self.__threshold = ignore_thres
		self.__mse = torch.nn.MSELoss()
		self.__bce = torch.nn.BCELoss()

	def forward(self, outputs, targets):
		pass

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
	for batch_idx in range(targets.size(0)):
		g_xy = targets[batch_idx][...,:2].squeeze(0) * g
		g_wh = targets[batch_idx][...,2:4].squeeze(0) * g
		ious = torch.stack([iou_wh(x, g_wh) for x in anchors])
		indices = ious.max(0)[1]
		mask[batch_idx, indices, g_xy[0].long(), g_xy[1].long()] = True
		tx[batch_idx, indices, g_xy[0].long(), g_xy[1].long()] = g_xy[0] - g_xy[0].floor()
		ty[batch_idx, indices, g_xy[0].long(), g_xy[1].long()] = g_xy[1] - g_xy[1].floor()
	return mask, tx, ty, tw, th, None
