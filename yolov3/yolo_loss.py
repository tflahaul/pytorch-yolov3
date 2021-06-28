from torchvision.ops._box_convert import _box_xyxy_to_cxcywh
from yolov3.configuration import CONFIG

import torch

class YOLOv3Loss(torch.nn.Module):
	def __init__(self) -> None:
		super(YOLOv3Loss, self).__init__()
		self.__mse = torch.nn.MSELoss()
		self.__bce = torch.nn.BCELoss()

	def __str__(self):
		return (f'x={self.metrics[0]:.2f}, y={self.metrics[1]:.2f}, '
			f'w={self.metrics[2]:.2f}, h={self.metrics[3]:.2f}, '
			f'obj={self.metrics[4]:.2f}, cls={self.metrics[5]:.2f}')

	def __anchors_iou(self, anchor, boxes):
		inter = torch.min(anchor[0], boxes[:, 0]) * torch.min(anchor[1], boxes[:, 1])
		union = (anchor[0] * anchor[1]) + (boxes[:, 0] * boxes[:, 1]) - inter
		return inter / (1e-8 + union)

	@torch.no_grad()
	def __build_targets(self, output_shape, anchors, targets):
		b, a, g, g, _ = output_shape # batch, anchors, grid size
		no_mask = torch.ones((b, a, g, g), dtype=torch.bool)
		mask = torch.zeros((b, a, g, g), dtype=torch.bool)
		tcls = torch.zeros((b, a, g, g, CONFIG.classes))
		xywh = torch.zeros((4, b, a, g, g))
		sample = targets[:, 0].long()
		boxes = _box_xyxy_to_cxcywh(targets[:, 1:5]) * g
		gx, gy = boxes[:, :2].t().long()
		ious = torch.stack([self.__anchors_iou(x, boxes[:, 2:4]) for x in anchors])
		_, best = ious.max(0)
		xywh[0, sample, best, gy, gx] = boxes[:, 0] - boxes[:, 0].floor()
		xywh[1, sample, best, gy, gx] = boxes[:, 1] - boxes[:, 1].floor()
		xywh[2, sample, best, gy, gx] = torch.log(1e-8 + (boxes[:, 2] / anchors[best, 0]))
		xywh[3, sample, best, gy, gx] = torch.log(1e-8 + (boxes[:, 3] / anchors[best, 1]))
		tcls[sample, best, gy, gx, targets[:, -1].long()] = 1.0
		no_mask[sample, best, gy, gx] = False
		mask[sample, best, gy, gx] = True
		for idx, item in enumerate(ious.t()):
			no_mask[sample[idx], item > CONFIG.confidence_thres, gy[idx], gx[idx]] = False
		return mask, no_mask, xywh, tcls

	def forward(self, outputs, targets):
		self.metrics = torch.zeros(6, requires_grad=False)
		for output, prediction, anchors in outputs:
			output = output.cpu()
			prediction = prediction.cpu()
			mask, no_mask, txywh, classes = self.__build_targets(prediction.shape, anchors, targets)
			self.metrics[0] += self.__mse(output[..., 0][mask], txywh[0][mask])
			self.metrics[1] += self.__mse(output[..., 1][mask], txywh[1][mask])
			self.metrics[2] += self.__mse(output[..., 2][mask], txywh[2][mask])
			self.metrics[3] += self.__mse(output[..., 3][mask], txywh[3][mask])
			obj_loss = self.__bce(output[..., 4][mask], mask.float()[mask])
			noobj_loss = self.__bce(output[..., 4][no_mask], mask.float()[no_mask])
			self.metrics[4] += (obj_loss + (100.0 * noobj_loss))
			self.metrics[5] += self.__bce(output[..., 5:][mask], classes[mask])
		return self.metrics.sum()
