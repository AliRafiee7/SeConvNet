# SeConvNet
 
**A convolutional neural network using selective convolutional blocks for very high density salt-and-pepper noise removal in gray-scale and color images**

## Requirements
TensorFlow <img align="left" height="25" src="figs/tf.png">

NumPy <img align="left" height="25" src="figs/numpy.jpg">

OpenCV <img align="left" height="25" src="figs/OpenCV.png">

scikit-image <img align="left" height="25" src="figs/scikit-image.png">

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

## Pre-trained Model
The pre-trained weights are stored in the *weights* folder.

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
Filename of the model's weights. The default is *model_050.hdf5*.
- **test_dir** <br />
Path of test data containing test datasets. The default is *data/Test*.
- **dataset** <br />
Name of test dataset. The default is *BSD68*.
- **result_dir** <br />
Path of saving denoised images. The default is *results*.
- **save_result** <br />
Whether to save denoised images or not. It is *0* for not saving and *1* for saving. The default is *0*.

<!---## This repository contains the python codes for the implementation of the paper "[A convolutional neural network using selective convolutional blocks for very high density salt-and-pepper noise removal in gray-scale and color images](https://doi.org/10.1007/s12652-022-03747-7)".

Citation
Rafiee, A.A., Farhang, M. A convolutional neural network using selective convolutional blocks for very high density salt-and-pepper noise removal in gray-scale and color images. *Journal* (2022). https://doi.org/

[Download citation](https://)


### DOI
https://doi.org/

## Abstract
... --->
