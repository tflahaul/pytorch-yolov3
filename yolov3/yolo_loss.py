from yolov3.configuration import CONFIG
import yolov3.utils as utils
import torch

class YOLOv3Loss(torch.nn.Module):
	def __init__(self) -> None:
		super(YOLOv3Loss, self).__init__()
		self.__mse = torch.nn.MSELoss()
		self.__bce = torch.nn.BCELoss()
		self.metrics = list([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	def __str__(self):
		return (f'x={self.metrics[0]:.2f}, y={self.metrics[1]:.2f}, '
			f'w={self.metrics[2]:.2f}, h={self.metrics[3]:.2f}, '
			f'obj={self.metrics[4]:.2f}, cls={self.metrics[5]:.2f}')

	@torch.no_grad()
	def __anchors_iou(self, box1, box2):
		inter = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
		union = (1e-8 + (box1[0] * box1[1])) + (box2[0] * box2[1]) - inter
		return inter / union

	@torch.no_grad()
	def __build_targets(self, outputs, anchors, targets):
		b, a, g, _, _ = outputs.shape # batch, anchors, grid size
		mask = torch.BoolTensor(b, a, g, g, device=CONFIG.device).fill_(False)
		no_mask = torch.BoolTensor(b, a, g, g, device=CONFIG.device).fill_(True)
		tx = torch.zeros(b, a, g, g, device=CONFIG.device)
		ty = torch.zeros(b, a, g, g, device=CONFIG.device)
		tw = torch.zeros(b, a, g, g, device=CONFIG.device)
		th = torch.zeros(b, a, g, g, device=CONFIG.device)
		tc = torch.zeros(b, a, g, g, CONFIG.classes, device=CONFIG.device)
		for batch_idx in range(targets.size(0)):
			for target in targets[batch_idx]:
				boxes = utils.xyxy2xywh(target[:4]) * g
				ious = torch.stack([self.__anchors_iou(x, boxes[2:4]) for x in anchors])
				value, best = ious.max(0)
				mask[batch_idx, best, int(boxes[1]), int(boxes[0])] = True
				no_mask[batch_idx, best, int(boxes[1]), int(boxes[0])] = False
				for index, item in enumerate(ious):
					if item > CONFIG.confidence_thres:
						no_mask[batch_idx, index, int(boxes[1]), int(boxes[0])] = False
				tx[batch_idx, best, int(boxes[1]), int(boxes[0])] = boxes[0] - boxes[0].floor()
				ty[batch_idx, best, int(boxes[1]), int(boxes[0])] = boxes[1] - boxes[1].floor()
				tw[batch_idx, best, int(boxes[1]), int(boxes[0])] = torch.log(boxes[2] / anchors[best][0])
				th[batch_idx, best, int(boxes[1]), int(boxes[0])] = torch.log(boxes[3] / anchors[best][1])
				tc[batch_idx, best, int(boxes[1]), int(boxes[0]), int(target[-1])] = 1.0
		return mask, no_mask, tx, ty, tw, th, tc

	def forward(self, outputs, targets):
		self.metrics = list([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
		for output, prediction, anchors in outputs:
			mask, no_mask, tx, ty, tw, th, tc = self.__build_targets(prediction, anchors, targets)
			self.metrics[0] += self.__mse(output[...,0][mask], tx[mask])
			self.metrics[1] += self.__mse(output[...,1][mask], ty[mask])
			self.metrics[2] += self.__mse(output[...,2][mask], tw[mask])
			self.metrics[3] += self.__mse(output[...,3][mask], th[mask])
			obj_loss = self.__bce(output[...,4][mask], mask.float()[mask])
			noobj_loss = self.__bce(output[...,4][no_mask], mask.float()[no_mask]) * 100.0
			self.metrics[4] += (obj_loss + noobj_loss)
			self.metrics[5] += self.__bce(output[...,5:][mask], tc[mask])
		return sum(self.metrics)
