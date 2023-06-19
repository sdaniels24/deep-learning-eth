# Context-Aware GAN with Feature Augmentation
Current child face generators are restricted by the limited size of the available datasets. In addition, feature selection can prove to be a significant challenge, especially due to the large amount of features that need to be trained for. To manage these problems, we proposed CADA-GAN, a Context-Aware GAN that allows optimal feature extraction, with added robustness from additional Data Augmentation. CADA-GAN is adapted from the popular StyleGAN2-Ada model, with attention on augmentation and segmentation of the parent images.

The project proposal and report can be found in the folder [Report](https://github.com/sdaniels24/deep-learning-eth/tree/main/Report).

## Installation
First, ensure you have an environment that functions with python=3.7.4. E.g create one in conda using:
```bash
conda create -n envname python=3.7.4
```

Download the repository and unzip in a location of your choice.
```bash
git clone https://github.com/sdaniels24/deep-learning-eth.git
```

Once finished, please install the following requirements file.
```bash
pip install -r requirement.txt
```
In case torch is not installed properly, try the following command:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

#### For image segmentation, please enter the following lines in your command prompt, starting from the working directory.
```bash
cd ImageSegmentation
cd face_parsing
git lfs pull
pip install -e .
git clone https://github.com/hhj1897/face_detection
cd face_detection
git lfs pull
pip install -e .
cd ../
git clone https://github.com/hhj1897/roi_tanh_warping
cd roi_tanh_warping
pip install -e .
cd ../../../
```
Note that the model weights for the pix2pix GAN have to be downloaded separately using the following link: https://polybox.ethz.ch/index.php/s/Kku0HvLrQS1UFC1.
You can also find the resnet/rtnet weights there, in case git lfs was unable to properly download them.

#### For image augmentation, you can enter the following lines in your command prompt.

The library can be installed from GitHub using `pip`.

For Linux and MacOS:
```bash
pip install 'git+https://github.com/sdaniels24/deep-learning-eth.git#egg=augmentations&subdirectory=ImageAugmentation'
```

For Windows:
```bash
pip install 'git+https://github.com/sdaniels24/deep-learning-eth.git#egg=augmentations^&subdirectory=ImageAugmentation'
```
For more details of the Image Augmentation, please check the Readme file inside the ImageAugmentation folder.

## GAN

You have to download and extract the folder 'pretrained' in the working directory. The folder can be downloaded from here: https://polybox.ethz.ch/index.php/s/Kku0HvLrQS1UFC1.


## Running

Example: run with augmentation (mixup) and color segmentation
```bash
python main.py --augment True --mixup True --segment 1 --transfer-learn True --model "ImageSegmentation/pix2pixGAN/models/model_seg0_256.h5" --epochs-lat 400
```


## Contributions:
- Sofie Daniëls: project proposal, project report, image segmentation, child face generation
- Jiugeng Sun: project proposal, project report, image augmentation
- Jiaxing Qie: project proposal, project report, child face generation
