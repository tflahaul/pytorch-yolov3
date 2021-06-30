## pytorch-yolov3
Yet another PyTorch implementation of YOLOv3 (https://pjreddie.com/darknet/yolo/) for object detection.

### Installation
Clone the repository using `git clone https://github.com/tflahaul/pytorch-yolov3`<br/>
Create a new virtual environment with `python -m venv ENV_NAME`<br/>
Activate the environment `source ENV_NAME/bin/activate`<br/>
Install the dependencies `python -m pip install -r requirements.txt`

### How to use
`config.yaml` is a configuration file where you can find/edit many useful parameters like the labels or number of training iterations.<br/>
`yolov3-config.json` contains the description of the network. DON'T CHANGE IT.<br/>
`training.py` and `detection.py` are training and detection scripts.

<br/>

`training.py` is the training script. It outputs a `.pth` file of roughly 240Mb that contains the weights for the model parameters.<br/>
This script can take multiple arguments but `dataset` is the only mandatory one. It is the path to your dataset folder, which should have the following architecture :

```bash
dataset
dataset/images/
dataset/targets/
```

Other arguments include `enable-cuda` to enable GPU acceleration on CUDA-capable devices and `load` to train an already trained model (transfer learning).

<br/>

`detection.py` isn't an actual program but was designed more like an API to simplify your detection scripts. You have to import it in your code like a module and you'll be able to use the functions like any regular API.<br/>

Here is an example :

```python
from detection import detect_from_single_image
from PIL import Image

import torchvision.transforms as tsfrm
import torchvision.utils as utils

model = torch.load('pytorch-entire-yolov3.pth')
img = tsfrm.ToTensor()(Image.open('nude.png').convert('RGB'))
boxes = detect_from_single_image(model, img)
img = utils.draw_bounding_boxes(tsfrm.ConvertImageDtype(torch.uint8)(img), boxes[...,:4])
utils.save_image(tsfrm.ConvertImageDtype(torch.float32)(img), 'nude_pred.png')
```
