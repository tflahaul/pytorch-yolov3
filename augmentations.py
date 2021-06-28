import torch

class HorizontalFlipWithTargets(torch.nn.Module):
	def __init__(self, p: float = 0.5) -> None:
		super(HorizontalFlipWithTargets, self).__init__()
		self.p = p

	def forward(self, image: torch.Tensor, targets: torch.Tensor):
		if torch.rand(1) < self.p:
			boxes = targets[:, 1:5]
			offset = torch.abs(boxes[:, 2] - boxes[:, 0])
			boxes[:, ::2] = 1.0 - boxes[:, ::2]
			boxes[:, 0] = boxes[:, 0] - offset
			boxes[:, 2] = boxes[:, 2] + offset
			targets[:, 1:5] = boxes
		return image.flip(-1), targets
