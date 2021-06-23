import torch

class HorizontalFlipWithTargets(torch.nn.Module):
	def __init__(self, p: float = 0.5) -> None:
		super(HorizontalFlipWithTargets, self).__init__()
		self.p = p

	def forward(self, image, targets):
		if torch.rand(1) < self.p:
			image = image.flip(-1)
			targets[:, 1:5:2] += 2 * ((image.size(-1) / 2) - targets[:, 1:5:2])
			off = torch.abs(targets[:, 1] - targets[:, 3])
			targets[:, 1] = targets[:, 1] - off
			targets[:, 3] = targets[:, 3] + off
		return image, targets
