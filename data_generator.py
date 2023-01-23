import numpy as np
import cv2
import glob


patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    img = cv2.imread(file_name, -1)
    h, w = img.shape[0:2]
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                # patches.append(x)
                # data aug
                for k in range(0, aug_times):
                    x = data_aug(x, mode=np.random.randint(0, 8))
                    #x = x.astype('float32')/255.0
                    patches.append(x)
    return patches


def data_gen(data_dir='data/Train', verbose=False):
    ext = ['png', 'jpg', 'gif', 'bmp']
    file_list = []
    [file_list.extend(glob.glob(data_dir + '/*.' + e)) for e in ext]
    data = []
    for i in range(len(file_list)):
        patch = gen_patches(file_list[i])
        data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data)
    if data.ndim==5:
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
    else:
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    discard_n = len(data) - len(data) // batch_size * batch_size;
    data = np.delete(data, range(discard_n), axis=0)
    return data