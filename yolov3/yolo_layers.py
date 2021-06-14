import torch

class RouteLayer(torch.nn.Module):
	def __init__(self, layers : list) -> None:
		super(RouteLayer, self).__init__()
		self.indexes = layers

class ShortcutLayer(torch.nn.Module):
	def __init__(self, from_index : int) -> None:
		super(ShortcutLayer, self).__init__()
		self.index = from_index

class YoloDetectionLayer(torch.nn.Module):
	def __init__(self, anchors, img_size, classes) -> None:
		super(YoloDetectionLayer, self).__init__()
		self.__bbox_attrs = (5 + classes) # (tx, ty, tw, th, obj, cls)
		self.__anchors = torch.Tensor(anchors)
		self.__nb_anchors = len(anchors)
		self.__imsize = img_size

	def __grid_offsets(self, gs : int):
		x = torch.arange(gs, device=self.__device).repeat(gs, 1).view(1, 1, gs, gs)
		y = torch.arange(gs, device=self.__device).repeat(gs, 1).t().view(1, 1, gs, gs)
		return torch.stack((x, y), -1)

	def forward(self, inputs):
		samples, _, y, x = inputs.shape
		stride = self.__imsize // y
		g = self.__imsize // stride
		self.out = inputs.view(samples, self.__nb_anchors, g, g, self.__bbox_attrs)
		self.scaled_anchors = self.__anchors.true_divide(stride)
		self.out[..., :2] = torch.sigmoid(self.out[..., :2])
		self.out[..., 4:] = torch.sigmoid(self.out[..., 4:])
		self.pred = self.out.detach().clone()
		self.pred[..., :2] = self.pred[..., :2] + self.__grid_offsets(g)
		self.pred[..., 2:4] = self.pred[..., 2:4].exp() * self.scaled_anchors.view(1, -1, 1, 1, 2)
		if not self.training:
			self.out = self.pred.view(samples, -1, self.__bbox_attrs)
			self.out[..., :4].mul_(stride)
		return self.out
