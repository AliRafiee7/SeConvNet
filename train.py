import argparse
import numpy as np
from tensorflow import keras
from tensorflow.math import reduce_sum, square
import os
from model import SeConvNet
from SPN import SPN
from data_generator import data_gen


parser = argparse.ArgumentParser()
parser.add_argument('--noise_density', default=0.95, type=float, help='noise density, should be in the interval [0, 1]')
parser.add_argument('--image_channels', default=1, type=int, help='number of channels; 1 for gray images and 3 for color images')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='initial learning rate for adam')
parser.add_argument('--train_dir', default='data/Train', type=str, help='path of train data')
parser.add_argument('--steps', dest='steps', type=int, default=2000, help='number of steps per epoch')
args = parser.parse_args()


color_mode = 'Gray' if args.image_channels == 1 else 'Color'


save_dir = os.path.join('weights', color_mode, 'SeConvNet_'+str(int(100*args.noise_density))) 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


model = SeConvNet(image_channels=args.image_channels)
model.summary


def train_datagen(epoch_iter=2000, epoch_num=5, batch_size=args.batch_size, data_dir=os.path.join(args.train_dir, color_mode)):
    while(True):
        n_count = 0
        if n_count == 0:
            xs = data_gen(data_dir)
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i+batch_size]]
                batch_x = batch_x.astype('float32')/255.0
                batch_y = SPN(batch_x, args.noise_density)
                batch_y[batch_y==1] = 0. 
                yield batch_y, batch_x 


def sum_squared_error(y_true, y_pred):
    return reduce_sum(square(y_pred - y_true))/2

model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss=sum_squared_error)


def scheduler(epoch):
    epochs=args.epoch
    initial_lr = args.lr
    if epoch<=int(0.7*epochs):
        lr = initial_lr
    else:
        lr = initial_lr/10 
    print('current learning rate is %1.8f' %lr)
    return lr

LearningRate_Scheduler = keras.callbacks.LearningRateScheduler(scheduler)

model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'), verbose=1, save_best_only=False, save_weights_only=True)

csv_logger = keras.callbacks.CSVLogger(os.path.join(save_dir,'training.log'), separator=",", append=True)

history = model.fit(train_datagen(batch_size=args.batch_size),steps_per_epoch=args.steps, epochs=args.epoch, verbose=1, callbacks=[model_checkpoint, csv_logger, LearningRate_Scheduler])