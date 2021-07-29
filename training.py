from yolov3.configuration import CONFIG
from yolov3.yolo_loss import YOLOv3Loss
from yolov3.network import Network
from argparse import ArgumentParser
from PIL import Image

import augmentations
import torchvision.transforms as tsfrm
import torch
import csv
import os

class DetectionDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_dir: str) -> None:
		super(DetectionDataset, self).__init__()
		dirs = sorted((d.path for d in os.scandir(dataset_dir) if d.is_dir()))
		self.__X = sorted((os.path.join(dirs[0], x) for x in os.listdir(dirs[0])))
		self.__y = sorted((os.path.join(dirs[1], x) for x in os.listdir(dirs[1])))
		assert len(self.__X) == len(self.__y), f'got {len(self.__X)} imgs but {len(self.__y)} targets'
		self.__default_transforms = tsfrm.Compose((
			tsfrm.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
			tsfrm.Resize((CONFIG.img_dim, CONFIG.img_dim), interpolation=tsfrm.InterpolationMode.LANCZOS),
			tsfrm.RandomGrayscale(p=0.15),
			tsfrm.ToTensor(),
			tsfrm.Normalize(mean=(0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201))))
		self.__targets_transforms = augmentations.CustomCompose((
			augmentations.RandomHorizontalFlip(p=0.1)))

	def __getitem__(self, index: int):
		bbox_attrs = list()
		img = self.__default_transforms(Image.open(self.__X[index]).convert('RGB'))
		with open(self.__y[index], mode='r', newline='') as fd:
			for ann in csv.reader(fd, quoting=csv.QUOTE_NONNUMERIC):
				bbox = torch.Tensor([[index] + ann[1:] + [CONFIG.labels.index(ann[0])]])
				bbox_attrs.append(bbox)
		return self.__targets_transforms(img, torch.cat(bbox_attrs, 0))

	def __len__(self) -> int:
		return len(self.__X)

def collate_batch_items(items: list):
	imgs = torch.stack((tuple(zip(*items)))[0])
	targets = torch.cat((tuple(zip(*items)))[1])
	targets[:, 0] = targets[:, 0] - targets[0, 0] # indices starts at 0
	return imgs, targets

def save_checkpoint(model, optimizer, scheduler) -> None:
	checkpoint = dict({
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler.state_dict()})
	torch.save(checkpoint, f'yolov3-chkpt-{scheduler.last_epoch}.pth')

def fit(model, optimizer, scheduler, dataset_dir: str) -> None:
	dataset = torch.utils.data.DataLoader(
		dataset=DetectionDataset(dataset_dir),
		batch_size=(CONFIG.batch_size // CONFIG.subdivisions),
		collate_fn=collate_batch_items,
		pin_memory=True,
		num_workers=4)
	criterion = YOLOv3Loss()
	model = model.train()
	for epoch in range(scheduler.last_epoch, CONFIG.epochs):
		running_loss = 0.0
		for minibatch, (images, targets) in enumerate(dataset):
			outputs = model(images.to(CONFIG.device, non_blocking=True))
			loss = criterion(outputs, targets)
			running_loss += loss.item()
			loss.backward()
			if (minibatch + 1) % CONFIG.subdivisions == 0:
				optimizer.step()
				optimizer.zero_grad()
		scheduler.step()
		print(f'epoch {(epoch + 1):>3d}/{CONFIG.epochs:<3d}, loss={running_loss:.6f}, {str(criterion)}')
		save_checkpoint(model, optimizer, scheduler)

def main() -> None:
	parser = ArgumentParser(description='YOLOv3 training script')
	parser.add_argument('dataset', type=str, help='dataset folder')
	parser.add_argument('--resume', type=str, metavar='checkpoint', help='resume from checkpoint')
	parser.add_argument('--enable-cuda', action='store_true', help='enable CUDA')
	parser.add_argument('--gpu', type=int, default=0, help='GPU index')
	arguments = parser.parse_args()
	if arguments.enable_cuda and torch.cuda.is_available() == True:
		setattr(CONFIG, 'device', torch.device('cuda:' + str(arguments.gpu)))
	torch.multiprocessing.set_start_method('spawn')
	model = Network().to(CONFIG.device)
	optimizer = torch.optim.AdamW(
		params=model.parameters(),
		weight_decay=CONFIG.decay,
		lr=CONFIG.learning_rate)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=CONFIG.epochs)
	if arguments.resume and os.path.exists(arguments.resume) == True:
		print(f'Loading checkpoint `{arguments.resume}`, this might take some time...')
		checkpoint = torch.load(arguments.resume, map_location=CONFIG.device)
		model.load_state_dict(checkpoint.get('model'))
		optimizer.load_state_dict(checkpoint.get('optimizer'))
		scheduler.load_state_dict(checkpoint.get('scheduler'))
	fit(model, optimizer, scheduler, arguments.dataset)
	torch.save(model.state_dict(), f'pytorch-yolov3-v{CONFIG.version}.pth')

if __name__ == '__main__':
	main()
