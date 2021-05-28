from yolov3.configuration import CONFIG
from argparse import ArgumentParser

import torchvision
import torchvision.transforms as tsfrm
import torchvision.io as io
import torch
import os

class ImagesDataset(torch.utils.data.Dataset):
	def __init__(self, images_dir) -> None:
		super(ImagesDataset, self).__init__()
		self.X = list([os.path.join(images_dir, item) for item in os.listdir(images_dir)])

	def __getitem__(self, index):
		img = io.read_image(self.X[index], io.image.ImageReadMode.RGB)
		img = tsfrm.Resize((CONFIG.img_dim, CONFIG.img_dim))(img)
		return img.to(CONFIG.device)

	def __len__(self):
		return len(self.X)

@torch.no_grad()
def detect(model_path, images_dir) -> None:
	dataset = torch.utils.data.DataLoader(
		dataset=ImagesDataset(images_dir),
		batch_size=CONFIG.batch_size)
	model = torch.load(model_path, map_location=torch.device(CONFIG.device))
	model.eval() # set bn layers to evaluation mode
	for image in dataset:
		outputs = model(tsfrm.ConvertImageDtype(torch.float32)(image))
		for idx, output in enumerate(outputs):
			for sample in range(output.size(0)):
				nms = torchvision.ops.nms(output[sample][...,:4], output[sample][...,4], CONFIG.iou_thres)
				output[:, nms[0], 2:4] += output[:, nms[0], :2] 
				print(f'{idx} - {output[:, nms[0], :4]}')
				img = torchvision.utils.draw_bounding_boxes(image[sample], output[:, nms[0], :4], ['dog'], ['red'], fill=True)
				torchvision.utils.save_image(tsfrm.ConvertImageDtype(torch.float32)(img), str(idx) + '_test.png')

def main() -> None:
	parser = ArgumentParser(description='Detect objects in images')
	parser.add_argument('images', type=str, help='Images dir')
	parser.add_argument('model', type=str, help='torch model')
	parser.add_argument('--gpu', type=int, choices=range(4), default=0, help='GPU index')
	parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
	arguments = parser.parse_args()
	if arguments.enable_cuda and torch.cuda.is_available() == True:
		setattr(CONFIG, 'device', 'cuda:' + str(arguments.gpu))
	detect(arguments.model, arguments.images)

if __name__ == '__main__':
	main()
