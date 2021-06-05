from torchvision.ops._box_convert import _box_cxcywh_to_xyxy
from yolov3.configuration import CONFIG
from argparse import ArgumentParser

import torchvision.transforms as tsfrm
import torchvision.io as io
import torchvision
import torch
import os

def get_bbox_attributes(boxes) -> list:
	available_colors = list(['red', 'blue', 'green'])
	attributes = list()
	for index in torch.argmax(boxes[...,5:], -1):
		attributes.append((CONFIG.labels[index], available_colors[index]))
	return list(zip(*attributes))

@torch.no_grad()
def detect(model_path, images_dir) -> None:
	dataset = list([os.path.join(images_dir, item) for item in os.listdir(images_dir)])
	model = torch.load(model_path, map_location=CONFIG.device)
	model.eval() # set bn layers to evaluation mode
	for image_name in dataset:
		img = io.read_image(image_name, io.image.ImageReadMode.RGB).to(CONFIG.device)
		img = tsfrm.Resize((CONFIG.img_dim, CONFIG.img_dim))(img)
		outputs = model(tsfrm.ConvertImageDtype(torch.float32)(img).unsqueeze(0))
		boxes = torch.cat([x[x[...,4] > CONFIG.confidence_thres] for x in outputs], 0)
		boxes[...,:4] = _box_cxcywh_to_xyxy(boxes[...,:4])
		boxes[...,:4] = torchvision.ops.clip_boxes_to_image(boxes[...,:4], (CONFIG.img_dim, CONFIG.img_dim))
		boxes = boxes[torchvision.ops.nms(boxes[...,:4], boxes[...,4], CONFIG.nms_thres)]
		bbox_attrs = get_bbox_attributes(boxes)
		img = torchvision.utils.draw_bounding_boxes(img, boxes[...,:4], labels=bbox_attrs[0], colors=bbox_attrs[1])
		torchvision.utils.save_image(tsfrm.ConvertImageDtype(torch.float32)(img), f'{image_name}_pred.png')
		print(f'Created {image_name}_pred.png')

def main() -> None:
	parser = ArgumentParser(description='detect objects using YOLOv3')
	parser.add_argument('model', type=str, help='path to model')
	parser.add_argument('images', type=str, help='images folder')
	parser.add_argument('--gpu', type=int, choices=range(4), default=0, help='GPU index')
	parser.add_argument('--enable-cuda', action='store_true', help='enable CUDA')
	arguments = parser.parse_args()
	if arguments.enable_cuda and torch.cuda.is_available() == True:
		setattr(CONFIG, 'device', torch.device('cuda:' + str(arguments.gpu)))
	detect(arguments.model, arguments.images)

if __name__ == '__main__':
	main()
