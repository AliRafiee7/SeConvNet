import numpy as np


def SPN(clear_img, density, salt_density=0.5):
    noisy_image = clear_img.copy()
    random1 = np.random.uniform(size=clear_img.shape)
    random2 = np.random.uniform(size=clear_img.shape)
    noisy_image[np.logical_and(random1 <= density, random2 <= salt_density)] = 0.
    noisy_image[np.logical_and(random1 <= density, random2 > salt_density)]  = 1.
    return noisy_image