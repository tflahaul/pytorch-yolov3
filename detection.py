from torchvision.ops._box_convert import _box_cxcywh_to_xyxy
from typing import Tuple

import torchvision.transforms
import torchvision.ops
import torch

IMG_SIZE = 416

__transformations = torchvision.transforms.Compose([
	torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
	torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1)])

def _box_resize(boxes: torch.Tensor, in_shape: Tuple[int, int], out_shape: Tuple[int, int]) -> torch.Tensor:
	resized = boxes.new(boxes.shape)
	resized[:, 0::2] = boxes[:, 0::2] * (out_shape[1] / in_shape[1])
	resized[:, 1::2] = boxes[:, 1::2] * (out_shape[0] / in_shape[0])
	return resized

@torch.no_grad()
def detect_from_single_image(
	model: torch.nn.Module,
	image: torch.Tensor,
	conf_thres: float = 0.5,
	nms_thres: float = 0.45,
) -> torch.Tensor:
	"""
	Performs a prediction over a single image.

	Args:
		model (Module): instance of model
		image (Tensor): image tensor of shape (3, H, W)
		conf_thres (float): objectness confidence threshold
		nms_thres (float): iou threshold used for non-maximum suppression

	Returns:
		Tensor[N, 85]: predicted boxes
	"""
	model.eval() # set bn layers to evaluation mode
	out = model(__transformations(image).unsqueeze(0))
	boxes = torch.cat([x[x[..., 4] > conf_thres] for x in out], 0).to(image.device)
	if boxes.size(0) > 0:
		boxes[..., :4] = _box_cxcywh_to_xyxy(boxes[..., :4])
		boxes = boxes[torchvision.ops.nms(boxes[..., :4], boxes[..., 4], nms_thres)]
		boxes[..., :4] = _box_resize(boxes[..., :4], (IMG_SIZE, IMG_SIZE), image.shape[-2:])
	return boxes

@torch.no_grad()
def detect_from_video_stream(
	model: torch.nn.Module,
	stream,
	conf_thres: float = 0.5,
	nms_thres: float = 0.45
) -> torch.Tensor:
	"""
	Performs a prediction over a video stream.

	Args:
		model (Module): instance of model
		stream (): stream of images of shape (3, H, W)
		conf_thres (float): objectness confidence threshold
		nms_thres (float): iou threshold used for non-maximum suppression

	Returns:
	"""
	model.eval() # set bn layers to evaluation mode
