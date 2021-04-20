from argparse import ArgumentParser
from yolov3.model import Network

from yolov3.configuration import CONFIG
import torch

def main() -> None:
	parser = ArgumentParser(description='YOLOv3 training script')
	parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
	parser.add_argument('--gpu', type=int, default=0, help='Index of the GPU to use')
	arguments = parser.parse_args()
	if not arguments.disable_cuda and torch.cuda.is_available() == True:
		setattr(CONFIG, 'device', 'cuda:' + str(arguments.gpu))
	else:
		setattr(CONFIG, 'device', 'cpu')
	model = Network().to(CONFIG.device)

if __name__ == '__main__':
	main()
