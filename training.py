from torchvision import transforms as tsfrm
from argparse import ArgumentParser

from yolov3.configuration import CONFIG
from yolov3.model import Network
from yolov3.yolo_loss import *
from PIL import Image

import torch
import csv
import os

__mse__ = torch.nn.MSELoss()
__bce__ = torch.nn.BCELoss()

class AnnotatedImagesDataset(torch.utils.data.Dataset):
	def __init__(self, images, targets) -> None:
		super(AnnotatedImagesDataset, self).__init__()
		self.X = sorted([os.path.join(images, item) for item in os.listdir(images)])
		self.y = sorted([os.path.join(targets, item) for item in os.listdir(targets)])
		assert len(self.X) == len(self.y)
		self.transform = tsfrm.Compose([
			tsfrm.Resize((CONFIG.img_height, CONFIG.img_width)),
			tsfrm.RandomGrayscale(p=0.1),
			tsfrm.ToTensor()])

	def __getitem__(self, index):
		bbox_attrs = list()
		img = self.transform(Image.open(self.X[index]).convert('RGB'))
		with open(self.y[index], mode='r', newline='') as fd:
			for ann in csv.reader(fd, quoting=csv.QUOTE_NONNUMERIC):
				label = torch.Tensor([CONFIG.labels.index(ann[0])])
				coords = torch.Tensor(ann[1:])
				bbox_attrs.append(torch.cat((coords, label), -1))
		return img.to(CONFIG.device), torch.stack(bbox_attrs).to(CONFIG.device)

	def __len__(self):
		return len(self.X)

def regressor(outputs, targets):
	losses = list()
	for output, anchors in outputs:
		mask, tx, ty, tw, th, tc = build_targets(output, anchors, targets)
		loss_x = __mse__(output[...,0][mask], tx[mask])
		loss_y = __mse__(output[...,1][mask], ty[mask])
		loss_c = __bce__(output[...,5:][mask], tc[mask])
#		loss_w = __mse__(output[...,2][mask], tw[mask]) ??
#		loss_h = __mse__(output[...,3][mask], th[mask])
		losses.append(loss_x + loss_y + loss_c)
	return sum(losses)

def fit(model, X, y, num_cpu) -> None:
	dataset = torch.utils.data.DataLoader(
		dataset=AnnotatedImagesDataset(X, y),
		batch_size=CONFIG.batch_size,
		num_workers=num_cpu)
	optimizer = torch.optim.AdamW(
		params=model.parameters(),
		weight_decay=CONFIG.decay,
		lr=CONFIG.learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer=optimizer,
		step_size=(CONFIG.epochs // 3),
		gamma=CONFIG.lr_decay)
	model.train()
	for epoch in range(CONFIG.epochs):
		running_loss = 0.0
		for images, targets in dataset:
			optimizer.zero_grad() # dont accumulate gradients
			outputs = model(images)
			loss = regressor(outputs, targets)
			loss.backward()
			optimizer.step()
			running_loss = running_loss + loss.item()
		print(f'epoch {(epoch + 1):>2d}/{CONFIG.epochs:>2d}, loss={running_loss:.6f}')
		scheduler.step()

def main() -> None:
	parser = ArgumentParser(description='YOLOv3 training script')
	parser.add_argument('images', type=str, help='Directory where images are stored')
	parser.add_argument('targets', type=str, help='Directory where targets are stored')
	parser.add_argument('--gpu', type=int, default=0, help='Index of GPU')
	parser.add_argument('--num-cpu', type=int, default=os.cpu_count(), help='Number of CPUs to use')
	parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
	arguments = parser.parse_args()
	if not arguments.disable_cuda and torch.cuda.is_available() == True:
		setattr(CONFIG, 'device', 'cuda:' + str(arguments.gpu))
	model = Network().to(CONFIG.device)
	fit(model, arguments.images, arguments.targets, arguments.num_cpu)
#	torch.save(model.state_dict(), 'yolov3.torch')

if __name__ == '__main__':
	main()