"""
detect example
"""

from yolov3.configuration import CONFIG
from detection import detect_from_single_image
from argparse import ArgumentParser
from PIL import Image

import torchvision.transforms as tsfrm
import torchvision
import torch
import os

def get_bbox_attributes(boxes) -> list:
	available_colors = list(['red', 'blue', 'green'])
	attributes = list()
	for index in torch.argmax(boxes[..., 5:], -1):
		attributes.append((CONFIG.labels[index], available_colors[index]))
	return list(zip(*attributes))

@torch.no_grad()
def detect(model_path, images_dir) -> None:
	dataset = [os.path.join(images_dir, item) for item in os.listdir(images_dir)]
	model = torch.load(model_path, map_location=CONFIG.device)
	for item in dataset:
		img = tsfrm.ToTensor()(Image.open(item).convert('RGB')).to(CONFIG.device)
		boxes = detect_from_single_image(model, img)
		bbox_attrs = get_bbox_attributes(boxes)
		img = torchvision.utils.draw_bounding_boxes(
			tsfrm.ConvertImageDtype(torch.uint8)(img),
			boxes[..., :4],
			labels=bbox_attrs[0],
			colors=bbox_attrs[1])
		torchvision.utils.save_image(tsfrm.ConvertImageDtype(torch.float)(img), f'{os.path.splitext(item)[0]}_pred.png')
		print(f'Created {os.path.splitext(item)[0]}_pred.png')

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
