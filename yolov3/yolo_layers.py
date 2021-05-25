import torch

class RouteLayer(torch.nn.Module):
	def __init__(self, layers) -> None:
		super(RouteLayer, self).__init__()
		self.indexes = layers

class ShortcutLayer(torch.nn.Module):
	def __init__(self, from_index) -> None:
		super(ShortcutLayer, self).__init__()
		self.index = from_index

class YoloDetectionLayer(torch.nn.Module):
	def __init__(self, anchors, img_dim, classes, device) -> None:
		super(YoloDetectionLayer, self).__init__()
		self.__bbox_attrs = (5 + classes) # (tx, ty, tw, th, objness, classes)
		self.__nb_anchors = len(anchors)
		self.__anchors = anchors
		self.__imsize = img_dim
		self.__device = device

	def __grid_offsets(self, gs):
		x = torch.arange(gs, device=self.__device).repeat(gs, 1).view(1, 1, gs, gs)
		y = torch.arange(gs, device=self.__device).repeat(gs, 1).t().view(1, 1, gs, gs)
		return torch.stack((x, y), -1)

	def forward(self, inputs):
		samples, _, ny, nx = inputs.shape
		stride = self.__imsize // ny
		g = self.__imsize // stride
		self.out = inputs.view(samples, self.__nb_anchors, g, g, self.__bbox_attrs)
		self.scaled_anchors = torch.Tensor([(aw / stride, ah / stride) for aw, ah in self.__anchors], device=self.__device)
		self.out[...,:2] = torch.sigmoid(self.out[...,:2]) + self.__grid_offsets(g)
		self.out[...,2] = torch.exp(self.out[...,2]) * self.scaled_anchors[:,0].view(1, self.__nb_anchors, 1, 1)
		self.out[...,3] = torch.exp(self.out[...,3]) * self.scaled_anchors[:,1].view(1, self.__nb_anchors, 1, 1)
		self.out[...,4:] = torch.sigmoid(self.out[...,4:])
		return self.out
