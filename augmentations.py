import torch

class RandomHorizontalFlip(torch.nn.Module):
	def __init__(self, p: float = 0.5) -> None:
		super(RandomHorizontalFlip, self).__init__()
		self.p = p

	def forward(self, image: torch.Tensor, targets: torch.Tensor):
		if torch.rand(1) < self.p:
			boxes = targets[:, 1:5]
			boxes[:, ::2] = 1.0 - boxes[:, [2, 0]]
			targets[:, 1:5] = boxes
		return image.flip(-1), targets
