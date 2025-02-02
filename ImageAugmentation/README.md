# AUgmentation

A python library that provides image augmentations for the Deep Learning Project of ETHz 2022. AUgmentations implements a modified version of the pytorch data loading and augmentation pipeline that allows for transformation of both the sample and target simultaneously. The three main blocks that can be directly plugged in are MixUp, AugMix, and SmartAugmentation, which are used specifically for augmenting the dataset of the Child Face Generation GAN.

## Dependencies

* torch - pytorch-gpu (or cpu)
* torchvision
* numpy
* matplotlib

Please make sure you install all the packages above before you install the **AU**gmentation Library.

## Installation

The library can be installed from GitHub using `pip`.

For Linux and MacOS:

```bash
pip install 'git+https://github.com/sdaniels24/deep-learning-eth.git#egg=augmentations&subdirectory=ImageAugmentation'
```

For Windows:

```bash
pip install 'git+https://github.com/sdaniels24/deep-learning-eth.git#egg=augmentations^&subdirectory=ImageAugmentation'
```

## Getting Started

After installing this library, it can be used directly in python scripts. The custom written dataset can be found under `augmentations.TSKinFace_Dataset` and the custom augmentations under `augmentations.augmentations`.

1. Example use for loading a dataset that uses MixUp:

```python
from augmentations import augmentations as A
from augmentations.TSKinFace_Dataset import TSKinDataset
from torchvision.datasets import ImageFolder

default_dataset = ImageFolder(
    root = 'root to the data folder',
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
)

dataset = TSKinDataset(
    root = 'root to the data folder',
    transform = [
        transforms.ToTensor(),
        A.MixUp(dataset=default_dataset)
    ]
)
```

2. Example use for loading a dataset that uses AugMix:

```python
from augmentations import augmentations as A
from augmentations.TSKinFace_Dataset import TSKinDataset

dataset = TSKinDataset(
    root = 'root to the data folder',
    transform = [
        transforms.ToTensor(),
        A.AugMix()
    ]
)
```

3. Example use for loading a dataset that uses P:

```python
dataset = TSKinDataset(
    root = 'root to the data folder',
    transform = [
        transforms.ToTensor(),
        A.P(A.MixUp(dataset=default_dataset), p=0.5),
        A.AugMix()
    ]
)
```

## Functions

### MixUp(dataset, alpha=0.2, min_lam=0.3, max_lam=0.7)

Function that is used to mix up the image of the parents. The function will generate a random number and mix the two images based on the random value.

**Parameters**:

`dataset`: The dataset used to mix the current image. The root that is used to construct the `dataset` should be the same as the TSKinDataset! Please refer to the `Example 1`.

`alpha`: Number used to generate the random number. Random number is retireved from `numpy.random.beta(alpha, alpha)`, which will draw the samples from a Beta distribution.

`min_lam` & `max_lam`: The minimum and maximum values that the random number could be. In order to have the valid values, please make sure the values are between 0 to 1 and make sure `min_lam` is smaller than `max_lam`.

### AugMix()

Function that is used to AugMix the image. The technique is probably not very helpful. Be careful when you want to use it.

### P(transform, p)

Besides two basic image manipulation techniques, we also implemented a P() function which can help you specify how likely you want to utilise the corresponding technique to the images. Please refer to `Example 3` if you want to use it.




## Notes

1. TSKinDataset along with the augmentation techniques is designed specifically for the TSKinFace_Data. Therefore, you can not use it in other dataset currently. 

2. Please pay attention that you need to load the dataset based on the TSKinDataset we defined but not the the ImageFolder from pytorch.

3. Please make sure you specify the right path to the dataset for `root`. Currently we only support you choose from `dataset/TSKinFace_Data/TSKinFace_cropped` (This may not be the right path you need!). There are three datasets which are `FMD`, `FMS`, and `FMSD`.

4. The image manipulation technique that we adopt is only working on the Parent side. There will not be any changes on the Child side.

## Making Changes

If you want to edit or add new augmentations or functionalities to this library, make the edit and then update the version number in [`setup.py`](setup.py#L8).
After the changes have been made and pushed to this repository, you can upgrade your version of `augmentations` by running the installation command again:

For Linux and MacOS:

```sh
pip install 'git+https://github.com/IvanSunjg/Advanced-Vision.git#egg=avgmentations&subdirectory=avgmentations'
```

