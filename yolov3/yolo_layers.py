import torch

class RouteLayer(torch.nn.Module):
	def __init__(self, layers_indexes) -> None:
		super(RouteLayer, self).__init__()
		self.indexes = layers_indexes

class ShortcutLayer(torch.nn.Module):
	def __init__(self, from_index) -> None:
		super(ShortcutLayer, self).__init__()
		self.index = from_index

class YoloDetectionLayer(torch.nn.Module):
	def __init__(self, anchors, num_classes, device) -> None:
		super(YoloDetectionLayer, self).__init__()
		self.__classes = num_classes
		self.__anchors = anchors
		self.__num_anchors = len(anchors)
		self.__device = device

	def forward(self, inputs):
		samples, _, ny, nx = inputs.shape
		x = inputs.view(samples, self.__num_anchors, (self.__classes + 5), ny, nx).permute(0, 1, 3, 4, 2).contiguous()
		print(f'out={x}')
		return x
