### Environment
- python 3
- pytorch 1.7.1  
- tensorboardX
- torchvision 0.8.2 
- yaml, numpy, tqdm, imageio, pillow, einops
## Quick Start

```
python demo.py --input xxx.png --model [MODEL_PATH] --resolution [HEIGHT],[WIDTH] --output output.png --gpu 0

e.g. python demo.py --input ../input.png --model ../save/ITSRN_last.pth --resolution 1560,1920 --output ../output.png --gpu 0

```
 `[MODEL_PATH]` denotes the `.pth` file)

python train.py --config xxxx.yaml --gpu xxx --name xxx
