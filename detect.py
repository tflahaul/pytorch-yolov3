from yolov3.configuration import CONFIG
from argparse import ArgumentParser

import yolov3.utils as utils
import torchvision.transforms as tsfrm
import torchvision.io as io
import torchvision
import torch
import os

class ImagesDataset(torch.utils.data.Dataset):
	def __init__(self, images_dir) -> None:
		super(ImagesDataset, self).__init__()
		self.data = list([os.path.join(images_dir, item) for item in os.listdir(images_dir)])

	def __getitem__(self, index):
		img = io.read_image(self.data[index], io.image.ImageReadMode.RGB)
		img = tsfrm.Resize((CONFIG.img_dim, CONFIG.img_dim))(img)
		return img.to(CONFIG.device)

	def __len__(self):
		return len(self.data)

@torch.no_grad()
def detect(model_path, images_dir) -> None:
	dataset = torch.utils.data.DataLoader(
		dataset=ImagesDataset(images_dir),
		batch_size=CONFIG.batch_size)
	model = torch.load(model_path, map_location=CONFIG.device)
	model.to(CONFIG.device)
	model.eval() # set bn layers to evaluation mode
	for images in dataset:
		outputs = model(tsfrm.ConvertImageDtype(torch.float32)(images))
		for index, scale_out in enumerate(outputs): # output has 3 different scales
			for sample in range(scale_out.size(0)):
				scale_out = scale_out[sample, scale_out[sample,:,4] > CONFIG.confidence_thres]
				scale_out[...,:4] = utils.xywh2xyxy(scale_out[...,:4])
				img = torchvision.utils.draw_bounding_boxes(images[sample], scale_out[...,:4], fill=True)
				torchvision.utils.save_image(tsfrm.ConvertImageDtype(torch.float32)(img), str(index) + '.png')

def main() -> None:
	parser = ArgumentParser(description='Detect objects in images')
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
