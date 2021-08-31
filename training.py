from yolov3.configuration import CONFIG
from yolov3.yolo_loss import YOLOv3Loss
from yolov3.network import Network
from argparse import ArgumentParser
from PIL import Image

import torchvision.transforms as tsfrm
import torch, csv, os
import augmentations

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
			tsfrm.RandomGrayscale(p=0.10),
			tsfrm.ToTensor(),
			tsfrm.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))))
		self.__targets_transforms = augmentations.CustomCompose((
			augmentations.RandomHorizontalFlip(p=0.03),))

	def __getitem__(self, index: int):
		bounding_boxes = list()
		img = self.__default_transforms(Image.open(self.__X[index]).convert('RGB'))
		with open(self.__y[index], mode='r', newline='') as fd:
			for ann in csv.reader(fd, quoting=csv.QUOTE_NONNUMERIC):
				bbox = torch.Tensor([[index] + ann[1:] + [CONFIG.labels.index(ann[0])]])
				bounding_boxes.append(bbox)
		return self.__targets_transforms(img, torch.cat(bounding_boxes, 0))

	def __len__(self) -> int:
		return len(self.__X)

def post_process_batch_items(items: list):
	imgs = torch.stack(tuple(zip(*items))[0])
	targets = torch.cat(tuple(zip(*items))[1])
	targets[:, 0] = targets[:, 0] - targets[0, 0] # indices starts at 0
	return imgs, targets

def save_checkpoint(model, optimizer, scheduler, epoch: int) -> None:
	checkpoint = dict({
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler.state_dict()})
	torch.save(checkpoint, f'yolov3-chkpt-{epoch}.pth')

def fit(model, optimizer, scheduler, dataset, checkpoint_steps: int) -> None:
	criterion = YOLOv3Loss()
	model = model.train()
	for epoch in range(scheduler.last_epoch, CONFIG.epochs):
		optimizer.zero_grad()
		running_loss = 0.0
		for minibatch, (images, targets) in enumerate(dataset):
			outputs = model(images.to(CONFIG.device))
			loss = criterion(outputs, targets)
			running_loss += loss.item()
			loss.backward()
			if (minibatch + 1) % CONFIG.subdivisions == 0:
				optimizer.step()
				optimizer.zero_grad()
				scheduler.step()
		optimizer.step() # for examples that doesn't fill the last batch
		print(f'epoch {(epoch + 1):>3d}/{CONFIG.epochs:<3d}, loss={running_loss:.6f}, {str(criterion)}')
		if (epoch + 1) % checkpoint_steps == 0:
			save_checkpoint(model, optimizer, scheduler, (epoch + 1))

def main() -> None:
	parser = ArgumentParser(description='YOLOv3 training script')
	parser.add_argument('dataset', type=str, help='dataset folder')
	parser.add_argument('--resume', type=str, metavar='checkpoint', help='resume from checkpoint')
	parser.add_argument('-s', '--step', type=int, default=1, help='step between checkpoints')
	parser.add_argument('--enable-cuda', action='store_true', help='enable CUDA')
	parser.add_argument('--gpu', type=int, default=0, help='GPU index')
	arguments = parser.parse_args()
	assert arguments.step > 0, 'step between checkpoints must be greater than 0'
	if arguments.enable_cuda and torch.cuda.is_available() == True:
		setattr(CONFIG, 'device', torch.device('cuda:' + str(arguments.gpu)))
	torch.multiprocessing.set_start_method('spawn')
	model = Network().to(CONFIG.device)
	dataset = torch.utils.data.DataLoader(
		dataset=DetectionDataset(dataset_dir=arguments.dataset),
		batch_size=(CONFIG.batch_size // CONFIG.subdivisions),
		collate_fn=post_process_batch_items,
		pin_memory=True,
		num_workers=4)
	optimizer = torch.optim.AdamW(
		params=model.parameters(),
		weight_decay=CONFIG.decay,
		lr=CONFIG.learning_rate)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=CONFIG.learning_rate,
		steps_per_epoch=len(dataset),
		epochs=CONFIG.epochs)
	if arguments.resume and os.path.exists(arguments.resume) == True:
		print(f'Loading checkpoint `{arguments.resume}`, this might take some time...')
		checkpoint = torch.load(arguments.resume, map_location=CONFIG.device)
		model.load_state_dict(checkpoint.get('model'))
		optimizer.load_state_dict(checkpoint.get('optimizer'))
		scheduler.load_state_dict(checkpoint.get('scheduler'))
	fit(model, optimizer, scheduler, dataset, arguments.step)
	torch.save(model.state_dict(), f'pytorch-yolov3-v{CONFIG.version}.pth')

if __name__ == '__main__':
	main()
