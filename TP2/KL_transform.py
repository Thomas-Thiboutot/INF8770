"""TP2 INF8770."""
import sys

from einops import rearrange
import numpy as np
from numpy import linalg as la
import cv2 as cv
import argparse


MAX_RGB = 255

def kl_transform(image_name: str, r: int, g: int, b: int):
    """Apply the KL transformation on png images.

    Args:
        image_name (str): The name of the image to transform

    Returns:
        _type_: Returns the image after compression and decompression
    """
    image = cv.imread('./data/{image}.png'.format(image=image_name))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.apply_along_axis(lambda channels: quantization(channels, [r, g, b]) , 2, image)
    rgb_mean = np.mean(image, axis=(0, 1), keepdims=True)
    vec_temp = image - np.squeeze(rgb_mean)
    vec_temp_rearranged = rearrange(vec_temp, 'h w c -> c (h w)')
    vec_prod_temp = np.dot(
        vec_temp_rearranged, np.transpose(vec_temp_rearranged),
    )
    rgb_mean_reshaped = np.squeeze(rgb_mean).reshape(3, 1)
    rgb_mean_temp = np.dot(rgb_mean_reshaped, np.transpose(rgb_mean_reshaped))
    cov_rgb = np.zeros((3, 3), dtype='double')
    cov_rgb = vec_prod_temp - rgb_mean_temp
    cov_rgb = cov_rgb / (image.shape[0] * image.shape[1])
    _, eigvec = la.eig(cov_rgb)
    eigvec = np.transpose(eigvec)
    eigvec_removed = np.copy(eigvec)
    axe_to_remove = np.argmin(la.norm(eigvec, axis=0))
    eigvec_removed[axe_to_remove, :] = [0, 0, 0]
    image_kl = np.dot(
        eigvec_removed,
        rearrange(np.subtract(vec_temp, rgb_mean), 'h w c -> c (h w)'),
    )
    inv_eigvec_removed = la.pinv(eigvec_removed)
    image_rebuilt = np.copy(image)
    image_rebuilt = np.dot(inv_eigvec_removed, image_kl)
    image_rebuilt = rearrange(
        image_rebuilt, 'c (h w) -> h w c', w=len(image[1]),
    )
    return np.clip(image_rebuilt, 0, MAX_RGB).astype('uint8')


def quantization(num: [int], nb_bits: [int]):
    """Quantize the pixel to a given number of bits.

    Args:
        num (int): The number to quantize
        nb_bits (int): The number of bits to quantized to the given number

    Returns:
        _type_: Returns the step for the correct number of bits
    """
    for i, channel in enumerate(num):
        step = (MAX_RGB + 1)/(2**nb_bits[i])
        num[i] = int(round(channel / step)*step)
    return num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='kl_transform',
        description='Applies a kl transformation to an png image',
        epilog='--------------------------------'
    )
    parser.add_argument('-i', '--image_name')
    parser.add_argument('-r', '--red', type=int)
    parser.add_argument('-g', '--green', type=int)
    parser.add_argument('-b', '--blue', type=int)
    args = parser.parse_args()
    kl_image = kl_transform(args.image_name, args.red, args.green, args.blue)
    cv.imshow('new_k1.png', kl_image)
    cv.waitKey(0)
    cv.destroyAllWindows()