import torch

def xyxy2xywh(box):
	new = box.new(box.shape)
	new[2] = torch.abs(box[2] - box[0])
	new[3] = torch.abs(box[3] - box[1])
	new[0] = box[0] + (new[2] / 2)
	new[1] = box[1] + (new[3] / 2)
	return new

def xywh2xyxy(boxes):
	new = boxes.new(boxes.shape)
	new[...,0] = boxes[...,0] - (boxes[...,2] / 2)
	new[...,1] = boxes[...,1] - (boxes[...,3] / 2)
	new[...,2] = boxes[...,0] + (boxes[...,2] / 2)
	new[...,3] = boxes[...,1] + (boxes[...,3] / 2)
	return new
