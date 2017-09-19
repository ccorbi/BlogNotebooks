from __future__ import print_function

from keras import backend as K
K.set_image_dim_ordering('th')  # ensure our dimension notation matches

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras import utils

import argparse
import os
import os.path
import time
import glob
import numpy as np

# Custom training data manager
from data import Data_Supplier
from serializer import coder_greys, combine_images, save_image


def generator_model():
    """
    deconvolution net from the latent space 100*1 to a 128*128*1 image
    by upsampling
    """
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 8 * 8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 8, 8), input_shape=(128 * 8 * 8,)))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    """
    Convolution net from  a  128*128*1 image to a 100*1 vector
    by downsampling
    """
    model = Sequential()
    model.add(Convolution2D(
        64, 5, 5,
        border_mode='same',
        input_shape=(1, 128, 128)))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    # We need this to train the generator using discirminator feedback
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def train(epochs, BATCH_SIZE, lr, reload_weights, data_folder='./train'):
    """
    Parameters
    ----------
    epochs: int
        Train for this many epochs

    BATCH_SIZE: int
        Size of minibatch

    lr : float
        Learning Rate

    weights: Bool
    if True, load weights from file, otherwise train the model from scratch.
    Use this if you have already saved state of the network and want to train it further.

    data_folder : str
        Path to the data
    """
    # init training

    training_data = Data_Supplier(
        path=data_folder, batch_size=BATCH_SIZE, encoder=coder_greys)
    training_data.shuffle()

    # init log output
    discr_loss = open('./discr_e{}_mb{}_lr{}.{}.out'.format(epochs,
                                                            BATCH_SIZE,
                                                            lr,
                                                            int(time.time())), 'w')
    gener_loss = open('./gener_e{}_mb{}_lr{}.{}.out'.format(epochs,
                                                            BATCH_SIZE,
                                                            lr,
                                                            int(time.time())), 'w')

    # init adversarial models
    discriminator = discriminator_model()
    generator = generator_model()

    if reload_weights:
        # restart training
        generator.load_weights('./Wgen_e{}_mb{}_lr{}.h5'.format(epochs,
                                                                BATCH_SIZE,
                                                                lr))
        discriminator.load_weights('./Wdisc_e{}_mb{}_lr{}.h5'.format(epochs,
                                                                BATCH_SIZE,
                                                                lr))

    # This is the feedback between models
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=lr, momentum=0.95, nesterov=True)
    g_optim = SGD(lr=lr, momentum=0.95, nesterov=True)

    generator.compile(loss='binary_crossentropy', optimizer="Adam")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    # do the training
    zdim = np.zeros((BATCH_SIZE, 100))
    for epoch in range(epochs):
        print("Epoch   {} - {}".format(epoch, time.ctime()))
        print("Batches {}".format(training_data.batches))
        # mini batches
        for index in range(training_data.batches):
            # latent vector ...
            for i in range(BATCH_SIZE):
                zdim[i, :] = np.random.uniform(-1, 1, 100)
            # load mini batch training elements
            image_batch = training_data.next_batch()
            # generate images using inception vector
            generated_images = generator.predict(zdim, verbose=0)
            # save images for monitor purpuses
            if not index % 5:
                image = combine_images(generated_images)
                name = str(epoch) + "_" + str(index) + ".png"
                dest_folder = os.path.normpath(os.getcwd() +
                                               "/generated-images_e{}_mb{}_lr{}/".format(epochs,
                                                                                        BATCH_SIZE,
                                                                                        lr))
                try:
                    # if folder exist fails in py27
                    os.makedirs(dest_folder)
                except:
                    pass

                save_image(image, name, folder=dest_folder)

            # put side by side generated and real
            X=np.concatenate((image_batch, generated_images))
            y=[1] * BATCH_SIZE + [0] * BATCH_SIZE
            # Train discriminator and get loss
            d_loss=discriminator.train_on_batch(X, y)
            print("%d\t%f" % (index, d_loss), file=discr_loss)

            # train the generator getting feedback from the discriminator
            for i in range(BATCH_SIZE):
                zdim[i, :]=np.random.uniform(-1, 1, 100)
            discriminator.trainable=False
            g_loss=discriminator_on_generator.train_on_batch(
                zdim, [1] * BATCH_SIZE)
            discriminator.trainable=True

            print("%d\t%f" % (index, g_loss), file=gener_loss)

            # save weights to restart training later
            if not epoch % 5:
                generator.save_weights('./Wgen_e{}_mb{}_lr{}.h5'.format(epochs,
                                                                        BATCH_SIZE,
                                                                        lr), True)
                discriminator.save_weights('./Wdisc_e{}_mb{}_lr{}.h5'.format(epochs,
                                                                        BATCH_SIZE,
                                                                        lr), True)
                #
                gener_loss.flush()
                discr_loss.flush()
        # reset training
        training_data.reset()
        training_data.shuffle()


def generate(num, generative_weights_file):
    # load generator
    generator=generator_model()
    generator.compile(loss = 'binary_crossentropy', optimizer = "SGD")
    # load trained weights
    generator.load_weights(generative_weights_file)
    # init latent vector with a random gaussian distrb.
    zdim=np.zeros((num, 100))
    for i in range(num):
        zdim[i, :]=np.random.uniform(-1, 1, 100)

    # create images
    generated_images=generator.predict(zdim, verbose = 1)

    for idx, image in enumerate(generated_images):
        image=image[0]
        save_image(image, name = "image_{}_{}.png".format(
            idx, generative_weights_file.replave('.h5', '')))



def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 250)
    parser.add_argument("--batch_size", type = int, default = 150)
    parser.add_argument("--lr", type = float, default = .0001)
    parser.add_argument("--reset", dest = "reset", action = "store_true")
    parser.add_argument("--verbose", dest = "verbose", action = "store_true")

    parser.set_defaults(reset = False, verbose = False)
    args=parser.parse_args()

    return args


if __name__ == '__main__':

    args=get_args()

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    print(time.ctime())
    # epochs, BATCH_SIZE, reload weights
    train(args.epochs, args.batch_size, args.lr, reload_weights = args.reset)
    # generate(5)
    print(time.ctime())
