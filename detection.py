from torchvision.ops._box_convert import _box_cxcywh_to_xyxy
from yolov3.network import Network
from typing import Optional, Tuple
from PIL import Image

import torchvision.transforms as trsfm
import torchvision.ops
import torch
import os

class __DetectionParameters(object):
	img_size: int = 416
	device: torch.device = torch.device('cpu')

CONFIG = __DetectionParameters()

__transformations = trsfm.Compose((
	trsfm.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
	trsfm.Resize((CONFIG.img_size, CONFIG.img_size), interpolation=trsfm.InterpolationMode.LANCZOS),
	trsfm.ToTensor(),
	trsfm.Normalize(mean=(0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201))))

def set_detection_parameters(
	img_size: Optional[int] = None,
	device: Optional[torch.device] = None
) -> None:
	if img_size is not None and img_size > 0:
		CONFIG.img_size = img_size
	if device is not None:
		CONFIG.device = device

def saved_model(filename: Optional[str] = None) -> torch.nn.Module:
	"""
	Instantiate a new model from a checkpoint or state_dict.

	Args:
		filename (str): location of the checkpoint/state_dict
	"""
	net = Network().to(CONFIG.device)
	if filename and os.path.exists(filename) == True:
		model = torch.load(filename, map_location=CONFIG.device)
		net.load_state_dict(model.get('model', model))
	return net

def _box_resize(boxes: torch.Tensor, in_shape: Tuple[int, int], out_shape: Tuple[int, int]) -> torch.Tensor:
	resized = boxes.new(boxes.shape)
	resized[:, 0::2] = boxes[:, 0::2] * (out_shape[1] / in_shape[1])
	resized[:, 1::2] = boxes[:, 1::2] * (out_shape[0] / in_shape[0])
	return resized

@torch.no_grad()
def detect_from_single_image(
	image: Image.Image,
	model: torch.nn.Module,
	conf_thres: float = 0.5,
	nms_thres: float = 0.45
) -> torch.Tensor:
	"""
	Performs a prediction over a single image.

	Args:
		image (Tensor): image tensor of shape (3, H, W)
		model (Module): instance of model
		conf_thres (float): objectness confidence threshold
		nms_thres (float): iou threshold used for non-maximum suppression

	Returns:
		Tensor[N, 85]: predicted boxes in (x1, y1, x2, y2, objness, cls) format
	"""
	model = model.eval() # set bn layers to evaluation mode
	out = model(__transformations(image).unsqueeze(0).to(CONFIG.device))
	boxes = torch.cat([x[x[..., 4] > conf_thres] for x in out], 0)
	if boxes.size(0) > 0:
		boxes[..., :4] = _box_cxcywh_to_xyxy(boxes[..., :4])
		boxes = boxes[torchvision.ops.nms(boxes[..., :4], boxes[..., 4], nms_thres)]
		boxes[..., :4] = _box_resize(boxes[..., :4], (CONFIG.img_size, CONFIG.img_size), image.size[::-1])
	return boxes

@torch.no_grad()
def detect_from_video_stream(
	stream,
	model: torch.nn.Module,
	conf_thres: float = 0.5,
	nms_thres: float = 0.45
) -> torch.Tensor:
	"""
	Performs a prediction over a video stream.

	Args:
		stream: stream of images, each of shape (3, H, W)
		model (Module): instance of model
		conf_thres (float): objectness confidence threshold
		nms_thres (float): iou threshold used for non-maximum suppression
	"""
	model = model.eval() # set bn layers to evaluation mode
