import functools
import torch

class RouteLayer(torch.nn.Module):
	def __init__(self, layers: list) -> None:
		super(RouteLayer, self).__init__()
		self.indexes = layers

class ShortcutLayer(torch.nn.Module):
	def __init__(self, from_index: int) -> None:
		super(ShortcutLayer, self).__init__()
		self.index = from_index

class YoloDetectionLayer(torch.nn.Module):
	def __init__(self, anchors, img_size, classes) -> None:
		super(YoloDetectionLayer, self).__init__()
		self.__bbox_attrs = (5 + classes) # (tx, ty, tw, th, obj, cls)
		self.__anchors = torch.Tensor(anchors)
		self.__nb_anchors = len(anchors)
		self.__imsize = img_size

	@functools.cache
	def __grid_offsets(self, size: int):
		grid = torch.arange(size).repeat(size, 1)
		return torch.stack((grid, grid.t()), -1).view(1, 1, size, size, 2)

	def forward(self, inputs):
		samples, _, y, x = inputs.shape
		stride = self.__imsize // y
		g = self.__imsize // stride
		self.out = inputs.view(samples, self.__nb_anchors, self.__bbox_attrs, y, x).permute(0, 1, 3, 4, 2)
		self.scaled_anchors = self.__anchors.true_divide(stride)
		self.out[..., :2] = torch.sigmoid(self.out[..., :2])
		self.out[..., 4:] = torch.sigmoid(self.out[..., 4:])
		self.pred = self.out.detach().clone()
		self.pred[..., :2] = self.pred[..., :2] + self.__grid_offsets(g)
		self.pred[..., 2:4] = self.pred[..., 2:4].exp() * self.scaled_anchors.view(1, -1, 1, 1, 2)
		if not self.training:
			self.out = self.pred.reshape(samples, -1, self.__bbox_attrs)
			self.out[..., :4].mul_(stride)
		return self.out
