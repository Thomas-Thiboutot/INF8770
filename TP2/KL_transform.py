"""TP2 INF8770."""
import sys

from einops import rearrange
import numpy as np
from numpy import linalg as la
import cv2 as cv


MAX_RGB = 255


def kl_transform(image_name: str):
    """Apply the KL transformation on png images.

    Args:
        image_name (str): The name of the image to transform

    Returns:
        _type_: Returns the image after compression and decompression
    """
    image = cv.imread('./data/{image}.png'.format(image=image_name))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
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
    kl_image = kl_transform(sys.argv[1])
