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
		self.__bbox_attrs = 5 + classes # (tx, ty, tw, th, objness, classes)
		self.__nb_anchors = len(anchors)
		self.__imgdim = img_dim[0]
		self.__anchors = anchors
		self.__device = device

	def __grid_offsets(self, grid_size):
		g = torch.arange(grid_size)
		return torch.meshgrid(g, g)

	def forward(self, inputs):
		samples, _, ny, nx = inputs.shape
		stride = self.__imgdim // ny
		self.g = self.__imgdim // stride
		self.x = inputs.view(samples, self.__bbox_attrs * self.__nb_anchors, self.g * self.g).transpose(1, 2).contiguous()
		self.x = self.x.view(samples, self.__nb_anchors * self.g * self.g, self.__bbox_attrs)
		anchors = torch.Tensor([(aw / stride, ah / stride) for aw, ah in self.__anchors], device=self.__device)
		xy_offsets = torch.cat(self.__grid_offsets(self.g), 1).repeat(1, self.__nb_anchors).view(-1, 2).unsqueeze(0)
		self.x[:,:,:2] = torch.sigmoid(self.x[:,:,:2]) + xy_offsets # x, y
		self.x[:,:,2:4] = torch.exp(self.x[:,:,2:4]) * anchors # w, h
		self.x[:,:,4:] = torch.sigmoid(self.x[:,:,4:]) # objectness and classes
		self.x[:,:,:4] *= stride
		return self.x
