## pytorch-yolov3
Yet another PyTorch implementation of YOLOv3 (https://pjreddie.com/darknet/yolo/) for object detection.

### Installation
Clone the repository using `git clone https://github.com/tflahaul/pytorch-yolov3`<br/>
If needed, create a new virtual environment with `python -m venv ENV_NAME`<br/>
Activate the environment `source ENV_NAME/bin/activate`<br/>
Install the dependencies `python -m pip install -r requirements.txt`

### How to use
`config.yaml` is a configuration file where you can find/edit many useful parameters like the labels or number of training iterations.<br/>
`yolov3-config.json` contains the description of the network. DON'T CHANGE IT.<br/>
`training.py` and `detection.py` are training and detection scripts.

<br/>

`training.py` is the training script. It outputs a `.pth` file of roughly 240Mb that contains the weights for the model parameters.<br/>
This script can take multiple arguments but `dataset` is the only mandatory one. It is the path to your dataset folder, which should have an architecture similar to this one:
```bash
dataset
dataset/images/img.png
dataset/targets/img.csv
```

The targets must be CSV files formatted as such:
```csv
"label_0",x1,y1,x2,y2
"label_1",x1,y1,x2,y2
```
`x1,y1` being the coordinates of the bounding box' upper left corner and `x2,y2` the lower right.<br/>

Other arguments include `enable-cuda` to enable GPU acceleration on CUDA-capable devices (single GPU only, multi-GPU training isn't supported yet) and `resume` to resume training from a checkpoint.

<br/>

`detection.py` isn't an actual program but was designed more like an API to simplify your detection scripts. You have to import it in your code like a module and you'll be able to use the functions like any regular API.<br/>

Here is an example (w/ torchvision==0.9.0):

```python
from detection import detect_from_single_image
from PIL import Image

import torchvision.transforms as tsfrm
import torchvision.utils as utils

model = torch.load('pytorch-yolov3.pth')
img = tsfrm.ToTensor()(Image.open('nude.png').convert('RGB'))
boxes = detect_from_single_image(model, img)
img = tsfrm.ConvertImageDtype(torch.uint8)(img)
img = utils.draw_bounding_boxes(img, boxes[...,:4])
img = tsfrm.ConvertImageDtype(torch.float32)(img)
utils.save_image(img, 'nude_pred.png')
```
