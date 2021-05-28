from yolov3.configuration import CONFIG
import torch

class YOLOv3Loss(torch.nn.Module):
	def __init__(self) -> None:
		super(YOLOv3Loss, self).__init__()
		self.__mse = torch.nn.MSELoss()
		self.__bce = torch.nn.BCELoss()
		self.metrics = list([0.0, 0.0, 0.0, 0.0, 0.0])

	def __str__(self):
		return (f'x={self.metrics[0]:.2f}, y={self.metrics[1]:.2f}, '
			f'w={self.metrics[2]:.2f}, h={self.metrics[3]:.2f}, '
			f'class={self.metrics[4]:.2f}')

	@torch.no_grad()
	def __iou_wh(self, box1, box2):
		inter = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
		union = (1e-8 + (box1[0] * box1[1])) + (box2[0] * box2[1]) - inter
		return inter / union

	def __build_targets(self, outputs, anchors, targets):
		b, a, g, _, _ = outputs.shape # batch, anchors, grid size
		mask = torch.BoolTensor(b, a, g, g, device=CONFIG.device).fill_(False)
		tx = torch.zeros(b, a, g, g, device=CONFIG.device)
		ty = torch.zeros(b, a, g, g, device=CONFIG.device)
		tw = torch.zeros(b, a, g, g, device=CONFIG.device)
		th = torch.zeros(b, a, g, g, device=CONFIG.device)
		tc = torch.zeros(b, a, g, g, CONFIG.classes, device=CONFIG.device)
		for batch_idx in range(targets.size(0)):
			for target in targets[batch_idx]:
				g_xy = target[:2] * g
				g_wh = (torch.abs(target[2:4] - target[:2]) * CONFIG.img_dim) * g
				ious = torch.stack([self.__iou_wh(x, g_wh) for x in anchors])
				value, best = ious.max(0)
				mask[batch_idx, best, int(g_xy[0]), int(g_xy[1])] = True
				tx[batch_idx, best, int(g_xy[0]), int(g_xy[1])] = g_xy[0] - g_xy[0].floor()
				ty[batch_idx, best, int(g_xy[0]), int(g_xy[1])] = g_xy[1] - g_xy[1].floor()
				tw[batch_idx, best, int(g_xy[0]), int(g_xy[1])] = torch.log(g_wh[0] / anchors[best][0])
				th[batch_idx, best, int(g_xy[0]), int(g_xy[1])] = torch.log(g_wh[1] / anchors[best][1])
				tc[batch_idx, best, int(g_xy[0]), int(g_xy[1]), int(target[-1])] = 1.0
		return mask, tx, ty, tw, th, tc

	def forward(self, outputs, targets):
		self.metrics = list([0.0, 0.0, 0.0, 0.0, 0.0])
		for output, anchors in outputs:
			mask, tx, ty, tw, th, tc = self.__build_targets(output, anchors, targets)
			self.metrics[0] += self.__mse(output[...,0][mask], tx[mask])
			self.metrics[1] += self.__mse(output[...,1][mask], ty[mask])
			self.metrics[2] += self.__mse(output[...,2][mask], tw[mask])
			self.metrics[3] += self.__mse(output[...,3][mask], th[mask])
			self.metrics[4] += self.__bce(output[...,5:][mask], tc[mask])
		return sum(self.metrics)
