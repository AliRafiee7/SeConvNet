# SeConvNet
 
## [A deep convolutional neural network for salt-and-pepper noise removal using selective convolutional blocks](https://doi.org/10.1016/j.asoc.2023.110535)
This repository contains the Python codes for the implementation of the paper "[A deep convolutional neural network for salt-and-pepper noise removal using selective convolutional blocks](https://doi.org/10.1016/j.asoc.2023.110535)."

## Citation
Rafiee, A.A., Farhang, M. A deep convolutional neural network for salt-and-pepper noise removal using selective convolutional blocks. *Applied Soft Computing* (2023). https://doi.org/10.1016/j.asoc.2023.110535

[Download citation as RIS](https://www.sciencedirect.com/sdfe/arp/cite?pii=S1568494623005537&format=application%2Fx-research-info-systems&withabstract=true)<br />
[Download citation as BibTeX](https://www.sciencedirect.com/sdfe/arp/cite?pii=S1568494623005537&format=text%2Fx-bibtex&withabstract=true)


### DOI
https://doi.org/10.1016/j.asoc.2023.110535

## Abstract

<p align="justify">
In recent years, there has been an unprecedented upsurge in applying deep learning approaches, specifically convolutional neural networks (CNNs), to solve image denoising problems, owing to their superior performance. However, CNNs mostly rely on Gaussian noise, and there is a conspicuous lack of exploiting CNNs for salt-and-pepper (SAP) noise reduction. In this paper, we proposed a deep CNN model, namely SeConvNet, to suppress SAP noise in gray-scale and color images. To meet this objective, we introduce a new selective convolutional (SeConv) block. SeConvNet is compared to state-of-the-art SAP denoising methods using extensive experiments on various common datasets. The results illustrate that the proposed SeConvNet model effectively restores images corrupted by SAP noise and surpasses all its counterparts at both quantitative criteria and visual effects, especially at high and very high noise densities.
</p>


## MSCF Architecture

![SeConvNet Architecture](/figs/SeConvNet.png)

## Requirements

<a href="##"><img alt="Python" align="left" height="20" width="20" src="figs/python.jpg"></a> [![Python-3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](##)

<a href="##"><img alt="TensorFlow" align="left" height="20" width="20" src="figs/tf.png"></a> [![TensorFlow-2.4.1](https://img.shields.io/badge/TensorFlow-2.4.1-blue.svg)](##)

<a href="##"><img alt="NumPy" align="left" height="20" width="20" src="figs/numpy.jpg"></a> [![NumPy-1.19.2](https://img.shields.io/badge/NumPy-1.19.2-blue.svg)](##)

<a href="##"><img alt="OpenCV" align="left" height="20" width="20" src="figs/OpenCV.png"></a> [![OpenCV-4.5.1](https://img.shields.io/badge/OpenCV-4.5.1-blue.svg)](##)

<a href="##"><img alt="scikit-image" align="left" height="20" width="20" src="figs/scikit-image.png"></a> [![scikit-image-0.19.2](https://img.shields.io/badge/scikit--image-0.19.2-blue.svg)](##)

<!---Matplotlib <img align="left" height="25" src="figs/matplotlib.png"> --->

## Training
Use this command to train the SeConvNet. You can change options (arguments) to your desired ones.
```
$ python train.py --noise_density 0.95 --image_channels 1 --epoch 50 --batch_size 128 --lr 1e-3 --train_dir data/Train --steps 2000
```
### Arguments
- **noise_density** <br />
Noise density for salt-and-pepper noise. It should be in the interval [0, 1]. The default is *0.95*.
- **image_channels** <br />
Number of channels in noisy images. It is *1* for gray images and *3* for color images. The default is *1*.
- **epoch** <br />
Number of epochs in training. The default is *50*.
- **batch_size** <br />
Number of batches in training. The default is *1*.
- **lr** <br />
Initial learning for Adam optimizer. The default is *0.001*.
- **train_dir** <br />
Path of training data. The default is *data/Train*.
- **steps** <br />
Number of steps per epoch in training. The default is *2000*.

## Testing
Use this command to test the SeConvNet. You can change options (arguments) to your desired ones.
```
$ python test.py --noise_density 0.95 --image_channels 1 --model_name model_050.hdf5 --test_dir data/Test --dataset BSD68 --result_dir results --save_result 0
```
### Arguments
- **noise_density** <br />
Noise density for salt-and-pepper noise. It should be in the interval [0, 1]. The default is *0.95*.
- **image_channels** <br />
Number of channels in noisy images. It is *1* for gray images and *3* for color images. The default is *1*.
- **model_name** <br />
Filename of the model's weights. The default is *model_050.hdf5*. You can use the trained weights which are available in the *weights* folder.
- **test_dir** <br />
Path of test data containing test datasets. The default is *data/Test*.
- **dataset** <br />
Name of test dataset. The default is *BSD68*.
- **result_dir** <br />
Path of saving denoised images. The default is *results*.
- **save_result** <br />
Whether to save denoised images or not. It is *0* for not saving and *1* for saving. The default is *0*.
