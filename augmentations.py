import torch

class CustomCompose(torch.nn.Module):
	def __init__(self, transforms: list) -> None:
		super(CustomCompose, self).__init__()
		self.__transforms = transforms

	def forward(self, image: torch.Tensor, targets: torch.Tensor):
		for trsfm in self.__transforms:
			image, targets = trsfm(image, targets)
		return image, targets

class RandomHorizontalFlip(torch.nn.Module):
	def __init__(self, p: float = 0.5) -> None:
		super(RandomHorizontalFlip, self).__init__()
		self.p = p

	def forward(self, image: torch.Tensor, targets: torch.Tensor):
		if torch.rand(1) < self.p:
			image = image.flip(-1)
			boxes = targets[:, 1:5]
			boxes[:, ::2] = 1.0 - boxes[:, [2, 0]]
			targets[:, 1:5] = boxes
		return image, targets

class RandomRotate(torch.nn.Module):
	def __init__(self, p: float = 0.5):
		super(RandomRotate, self).__init__()
		self.p = p

	def forward(self, image: torch.Tensor, targets: torch.Tensor):
		if torch.rand(1) < self.p:
			image.transpose_(-1, -2).flip(-1)
			boxes = targets[:, 1:5]
#			rotate boxes here
			targets[:, 1:5] = boxes
		return image, targets

class RandomGaussianNoise(torch.nn.Module):
	def __init__(self, p: float = 0.5, intensity: float = 0.25) -> None:
		super(RandomGaussianNoise, self).__init__()
		self.std = intensity
		self.p = p

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		if torch.rand(1) < self.p:
			noise = torch.normal(0, self.std, size=image.shape[1:]).repeat(image.size(0), 1, 1)
			image = image + noise
		return image
