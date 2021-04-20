from yolov3.yolo_layers import RouteLayer, ShortcutLayer, YoloDetectionLayer
from yolov3.configuration import CONFIG

import json, sys
import torch

def convolutional(item, out_filters):
	x = torch.nn.Sequential()
	x.add_module('conv', torch.nn.Conv2d(
		in_channels=out_filters[-1],
		out_channels=item.get('filters'),
		kernel_size=item.get('size'),
		stride=item.get('stride'),
		padding=(item.get('size') > 1),
		bias=(not item.get('batch_normalize'))))
	if item.get('batch_normalize', False) == True:
		x.add_module('bn', torch.nn.BatchNorm2d(
			num_features=item.get('filters'),
			momentum=CONFIG.bn_momentum))
	if item.get('activation', 'linear') == 'leaky':
		x.add_module('act', torch.nn.LeakyReLU(0.1))
	out_filters.append(item.get('filters'))
	return x

def upsample(item, _):
	return torch.nn.Upsample(scale_factor=item.get('stride'))

def shortcut(item, out_filters):
	out_filters.append(out_filters[1:][item.get('from')])
	return ShortcutLayer(item.get('from'))

def route(item, out_filters):
	out_filters.append(sum([out_filters[1:][x] for x in item.get('layers')]))
	return RouteLayer(item.get('layers'))

def yolo(item, _):
	return YoloDetectionLayer(
		anchors=list(map(item.get('anchors').__getitem__, item.get('mask'))),
		num_classes=CONFIG.classes,
		device=CONFIG.device)

def build_model_from_cfg():
	out_filters =  list([CONFIG.channels])
	with open(CONFIG.model, mode='r') as fd:
		for item in json.load(fd):
			yield getattr(sys.modules[__name__], item.get('type'))(item, out_filters)

class Network(torch.nn.Module):
	def __init__(self) -> None:
		super(Network, self).__init__()
		self.__outputs = list()
		self.__model = torch.nn.Sequential(*build_model_from_cfg())
		self.outputs = [x for x in self.__model if isinstance(x, YoloDetectionLayer)]
		for layer in self.__model.children():
			layer.register_forward_hook(lambda _, __, out : self.__outputs.append(out))

	def forward(self, inputs):
		for layer in self.__model:
			if isinstance(layer, RouteLayer):
				inputs = torch.cat([self.__outputs[idx] for idx in layer.indexes])
			elif isinstance(layer, ShortcutLayer):
				inputs = self.__outputs[-1] + self.__outputs[layer.index]
			elif isinstance(layer, YoloDetectionLayer):
				continue
			else:
				inputs = layer(inputs)
		self.__outputs.clear()
		return inputs
