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
	def __init__(self, anchors, num_classes) -> None:
		super(YoloDetectionLayer, self).__init__()
		self.__classes = num_classes
		self.__anchors = anchors
		print(self.__anchors)
