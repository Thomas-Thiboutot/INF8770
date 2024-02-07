"""TP2 INF8770."""
import sys

from einops import rearrange
import numpy as np
from matplotlib import pyplot as py
from numpy import linalg as la
import cv2 as cv


MAX_RGB = 255


def kl_transform(image_name: str):
    """Apply the KL transformation on png images.

    Args:
        image_name (str): The name of the image to transform
    """
    imagelue = cv.imread('./data/{image}.png'.format(image=image_name))
    imagelue = cv.cvtColor(imagelue, cv.COLOR_BGR2RGB)
    image = imagelue.astype('double')
    rgb_mean = np.mean(image, axis=(0, 1), keepdims=True)
    cov_rgb = np.zeros((3, 3), dtype='double')
    vec_temp = image - rgb_mean
    vec_temp_rearranged = rearrange(vec_temp, 'h w c -> c (h w)')
    vec_prod_temp = np.dot(
        vec_temp_rearranged, np.transpose(vec_temp_rearranged),
    )
    cov_rgb = np.add(cov_rgb, vec_prod_temp)
    cov_rgb = cov_rgb / (len(image)*len(image[0]))
    _, eigvec = la.eig(cov_rgb)
    eigvec = np.transpose(eigvec)
    eigvec_removed = np.copy(eigvec)
    axe_to_remove = np.argmin(la.norm(eigvec, axis=0))
    eigvec_removed[axe_to_remove, :] = [0, 0, 0]
    image_kl = np.dot(
        eigvec_removed,
        rearrange(np.subtract(vec_temp, rgb_mean), 'h w c -> c (h w)'),
    )
    # image_kl_r = rearrange(image_kl, 'c (h w) -> h w c', w=len(image[1]))
    inv_eigvec_removed = la.pinv(eigvec_removed)
    image_rebuilt = np.copy(image)
    image_rebuilt = np.dot(inv_eigvec_removed, image_kl)
    image_rebuilt = rearrange(
        image_rebuilt, 'c (h w) -> h w c', w=len(image[1]),
    )
    imageout = np.clip(image_rebuilt, 0, MAX_RGB)
    imageout = imageout.astype('uint8')
    print(imageout)


def quantization(num: int, nbbits: int):
    """Quantize the pixel to a given number of bits.

    Args:
        num (int): The number to quantize
        nbbits (int): The number of bits to quantized to the given number

    Returns:
        _type_: Returns the step for the correct number of bits
    """
    return round(num / (2**nbbits))


if __name__ == '__main__':
    kl_transform(sys.argv[1])
