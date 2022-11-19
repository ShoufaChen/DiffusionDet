## Getting Started with DiffusionDet



### Installation

The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN), and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
Thanks very much.

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install Detectron2 following https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation.

2. Prepare datasets
```
mkdir -p datasets/coco
mkdir -p datasets/lvis

ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017

ln -s /path_to_lvis_dataset/lvis_v1_train.json datasets/lvis/lvis_v1_train.json
ln -s /path_to_lvis_dataset/lvis_v1_val.json datasets/lvis/lvis_v1_val.json
```

3. Prepare pretrain models

DiffusionDet uses three backbones including ResNet-50, ResNet-101 and Swin-Base. The pretrained ResNet-50 model can be
downloaded automatically by Detectron2. We also provide pretrained
[ResNet-101](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/torchvision-R-101.pkl) and
[Swin-Base](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/swin_base_patch4_window7_224_22k.pkl) which are compatible with
Detectron2. Please download them to `DiffusionDet_ROOT/models/` before training:

```bash
mkdir models
cd models
# ResNet-101
wget https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/torchvision-R-101.pkl

# Swin-Base
wget https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/swin_base_patch4_window7_224_22k.pkl

cd ..
```

Thanks for model conversion scripts of [ResNet-101](https://github.com/PeizeSun/SparseR-CNN/blob/main/tools/convert-torchvision-to-d2.py)
and [Swin-Base](https://github.com/facebookresearch/Detic/blob/main/tools/convert-thirdparty-pretrained-model-to-d2.py).

4. Train DiffusionDet
```
python train_net.py --num-gpus 8 \
    --config-file configs/diffdet.coco.res50.yaml
```

5. Evaluate DiffusionDet
```
python train_net.py --num-gpus 8 \
    --config-file configs/diffdet.coco.res50.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

* Evaluate with arbitrary number (e.g 300) of boxes by setting `MODEL.DiffusionDet.NUM_PROPOSALS 300`.
* Evaluate with 4 refinement steps by setting `MODEL.DiffusionDet.SAMPLE_STEP 4`.


We also provide the [pretrained model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50_300boxes.pth)
of [DiffusionDet-300boxes](configs/diffdet.coco.res50.300boxes.yaml) that is used for ablation study.


### Inference Demo with Pre-trained Models
We provide a command line tool to run a simple demo following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/demo#detectron2-demo).

```bash
python demo.py --config-file configs/diffdet.coco.res50.yaml \
    --input image.jpg --opts MODEL.WEIGHTS diffdet_coco_res50.pth
```

We need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.
