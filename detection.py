from torchvision.ops._box_convert import _box_cxcywh_to_xyxy
from yolov3.configuration import CONFIG
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

def box_resize(boxes, in_shape, out_shape):
	resized_boxes = boxes.new(boxes.shape)
	resized_boxes[:, 0::2] = boxes[:, 0::2] * (out_shape[1] / in_shape[1])
	resized_boxes[:, 1::2] = boxes[:, 1::2] * (out_shape[0] / in_shape[0])
	return resized_boxes

@torch.no_grad()
def detect(model_path, images_dir) -> None:
	dataset = os.listdir(images_dir)
	model = torch.load(model_path, map_location=CONFIG.device)
	model.eval() # set bn layers to evaluation mode
	for item in dataset:
		img = tsfrm.ToTensor()(Image.open(os.path.join(images_dir, item)).convert('RGB')).to(CONFIG.device)
		outputs = model(tsfrm.Resize((CONFIG.img_dim, CONFIG.img_dim))(img).unsqueeze(0))
		boxes = torch.cat([x[x[..., 4] > CONFIG.confidence_thres] for x in outputs], 0)
		boxes[..., :4] = _box_cxcywh_to_xyxy(boxes[..., :4])
		boxes = boxes[torchvision.ops.nms(boxes[..., :4], boxes[..., 4], CONFIG.nms_thres)]
		boxes[..., :4] = box_resize(boxes[..., :4], (CONFIG.img_dim, CONFIG.img_dim), (img.size(-2), img.size(-1)))
		boxes[..., :4] = torchvision.ops.clip_boxes_to_image(boxes[..., :4], (img.size(-2), img.size(-1)))
		bbox_attrs = get_bbox_attributes(boxes)
		img = torchvision.utils.draw_bounding_boxes(
			image=tsfrm.ConvertImageDtype(torch.uint8)(img),
			boxes=boxes[..., :4],
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
