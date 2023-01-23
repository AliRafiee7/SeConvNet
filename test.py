import argparse
import os
import glob
import numpy as np
import cv2
from SPN import SPN
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from model import SeConvNet


parser = argparse.ArgumentParser()
parser.add_argument('--noise_density', default=0.95, type=float, help='noise density, should be in the interval [0, 1]')
parser.add_argument('--image_channels', default=1, type=int, help='number of channels; 1 for gray images and 3 for color images')
parser.add_argument('--model_name', default='model_050.hdf5', type=str, help='the model name')
parser.add_argument('--test_dir', default='data/Test', type=str, help='path of test data')
parser.add_argument('--dataset', default='BSD68', type=str, help='name of test dataset')
parser.add_argument('--result_dir', default='results', type=str, help='path of saving denoised images')
parser.add_argument('--save_result', default=1, type=int, help='save denoised images, 1 or 0')
args = parser.parse_args()


def to_tensor(img):
    if img.ndim==2:
        return np.expand_dims(np.expand_dims(img, -1), 0)
    else:
        return np.expand_dims(img, 0)
    

color_mode = 'Gray' if args.image_channels == 1 else 'Color'


model = SeConvNet(image_channels=args.image_channels)

weights_dir = os.path.join('weights', color_mode, 'SeConvNet_'+str(int(100*args.noise_density)), args.model_name)
model.load_weights(weights_dir)


img_format = ['png', 'jpg', 'gif', 'bmp']


files_dir = os.path.join(args.test_dir, args.dataset)
img_list = []
[img_list.extend(glob.glob('./' + files_dir + '/*.' + e)) for e in img_format]


MSEs = []
IEFs = []
PSNRs = []
SSIMs = []

for img in range(len(img_list)):
    y = cv2.imread(img_list[img], -1)
    y = y.astype('float32')/255.0
    
    x = SPN(y, args.noise_density)
    
    x_ = to_tensor(x)
    x_[x_==1] = 0. #convert 1 to 0
    x_hat = model.predict(x_)
    x_hat = np.squeeze(x_hat)
    

    save_dir = os.path.join(args.result_dir, args.dataset,'SeConvNet_'+str(int(100*args.noise_density)))
    if args.save_result:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        head, tail = os.path.split(img_list[img])
        cv2.imwrite(os.path.join(save_dir, tail), (x_hat* 255).astype(np.uint8))
        
    
    mse = mean_squared_error(y, x_hat)
    ief = mean_squared_error(y, x)/mean_squared_error(y, x_hat)
    psnr = peak_signal_noise_ratio(y, x_hat)
    if x_hat.ndim==3:
        ssim_0 = structural_similarity(y[:,:,0], x_hat[:,:,0])
        ssim_1 = structural_similarity(y[:,:,1], x_hat[:,:,1])
        ssim_2 = structural_similarity(y[:,:,2], x_hat[:,:,2])
        ssim = (ssim_0+ssim_1+ssim_2)/3
    else:
        ssim = structural_similarity(y, x_hat)

    MSEs.append(mse)
    IEFs.append(ief)
    PSNRs.append(psnr)
    SSIMs.append(ssim)

MSE = np.mean(MSEs)
IEF = np.mean(IEFs)
PSNR = np.mean(PSNRs)
SSIM = np.mean(SSIMs)

results = [MSE, IEF, PSNR, SSIM]
print('Dataset:', args.dataset, '\nNoise Density =', args.noise_density, '\nPSNR =', "%.2f" % PSNR, '\nSSIM =', "%.4f" % SSIM)