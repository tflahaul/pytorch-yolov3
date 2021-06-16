## pytorch-yolov3
Super simple PyTorch implementation of YOLOv3 (https://pjreddie.com/darknet/yolo/) for object detection and classification.

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

`training.py` is the training script. It outputs a `.pth` file of roughly 250Mb that contains the model parameters' weights.<br/>
This script can take multiple arguments but `dataset` is the only mandatory one. It is the path to your dataset folder, which should have the following architecture :

```bash
dataset
dataset/images/
dataset/targets/
```
Other arguments include `enable-cuda` to enable GPU acceleration on CUDA-capable devices and `load` to train an already trained model (transfer learning).

<br/>

`detection.py` isn't an actual program but a function.
