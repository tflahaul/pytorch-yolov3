from yolov3.configuration import CONFIG
from yolov3.yolo_layers import *

import json, sys
import torch

def convolutional(item, out_filters):
	x = torch.nn.Sequential()
	x.add_module('conv', torch.nn.Conv2d(
		in_channels=out_filters[-1],
		out_channels=item.get('filters'),
		kernel_size=item.get('size'),
		stride=item.get('stride'),
		padding=((item.get('size', 1) - 1)//2),
		bias=(not item.get('batch_normalize'))))
	if item.get('batch_normalize') == True:
		x.add_module('bn', torch.nn.BatchNorm2d(
			num_features=item.get('filters'),
			momentum=CONFIG.bn_momentum))
	if item.get('activation') == 'leaky':
		x.add_module('act', torch.nn.LeakyReLU(0.1))
	out_filters.append(item.get('filters'))
	return x

def upsample(item, out_filters):
	out_filters.append(out_filters[-1]) # repeat
	return torch.nn.Upsample(scale_factor=item.get('stride'))

def shortcut(item, out_filters):
	out_filters.append(out_filters[1:][item.get('from')])
	return ShortcutLayer(item.get('from'))

def route(item, out_filters):
	out_filters.append(sum((out_filters[1:][x] for x in item.get('layers'))))
	return RouteLayer(item.get('layers'))

def yolo(item, out_filters):
	out_filters.append(out_filters[-1]) # repeat
	return YoloDetectionLayer(
		anchors=[item.get('anchors')[i] for i in item.get('mask')],
		img_size=CONFIG.img_dim,
		classes=CONFIG.classes).to(CONFIG.device)

def build_model_from_cfg():
	filters =  list([CONFIG.channels])
	with open(CONFIG.model, mode='r') as fd:
		for item in json.load(fd):
			yield getattr(sys.modules[__name__], item.get('type'))(item, filters)

class Network(torch.nn.Module):
	def __init__(self) -> None:
		super(Network, self).__init__()
		self.__model = torch.nn.Sequential(*build_model_from_cfg())
		self.heads = [x for x in self.__model if isinstance(x, YoloDetectionLayer)]

	def forward(self, inputs):
		outputs = list()
		for module in self.__model.children():
			if isinstance(module, ShortcutLayer):
				inputs = outputs[-1] + outputs[module.index]
			elif isinstance(module, RouteLayer):
				inputs = torch.cat([outputs[i] for i in module.indexes], 1)
			else:
				inputs = module(inputs)
			outputs.append(inputs)
		if self.training == True:
			return list([(x.out, x.pred, x.scaled_anchors) for x in self.heads])
		return list([x.out for x in self.heads])
