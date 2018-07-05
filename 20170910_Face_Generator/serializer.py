import math
import numpy as np
from PIL import Image, ImageOps

def coder_greys(file_path):
    pixels = 128
    im = Image.open(file_path)
    im = ImageOps.fit(im, (pixels, pixels), Image.ANTIALIAS)
    im = ImageOps.grayscale(im)
    im = np.asarray(im)
    im = (im - 127.5) / 127.5

    return im


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[0, :, :]
    return image


def save_image(image, name, folder=False):

    if  folder:
        name = folder +'/'+ name
    # rescale to 0..255
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(name)
