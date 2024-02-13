"""TP2 INF8770."""
import os

import argparse
import shutil
import cv2 as cv
from einops import rearrange
import numpy as np
from numpy import linalg as la
from skimage.metrics import structural_similarity


DIR_LIST = os.listdir('./data')
MAX_RGB = 255


def kl_transform(image1_name: str,image2_name: str):
    """Apply the KL transformation on images.

    Args:
        image_name (str): The name of the image to transform
        r (int): The red channel number of bits
        g (int): The green channel number of bits
        b (int): The blue channel number of bits

    Returns:
        np.ndarray: Returns the image after compression and decompression
    """
    image1 = cv.imread('./data/{image}'.format(image=image1_name))
    image = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    
    mean = np.squeeze(np.mean(image, axis=(0,1), keepdims=True))
    
    cov = np.zeros((3,3), dtype = "double")
    vec_temp = image - mean
    vec_temp_reshaped = rearrange(vec_temp, 'h w c -> c (h w)')
    vec_prod_temp = np.dot(vec_temp_reshaped, np.transpose(vec_temp_reshaped))
    cov = cov + vec_prod_temp
    nb_pixels = image.shape[0] * image.shape[1]
    cov = cov / nb_pixels
    
    eigval, eigvec = la.eig(cov)
    eigvec = np.transpose(eigvec)
    eigvec_removed = np.copy(eigvec)
    eigvec_removed[np.argmin(eigval),:] = [0.0,0.0,0.0]
    
    image2 = cv.imread('./data/{image}'.format(image=image2_name))
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
    image2 = np.apply_along_axis(lambda channels: quantization(channels, [8, 8, 8]), 2, image2)
    mean2 = np.squeeze(np.mean(image2, axis=(0,1), keepdims=True))
    vec_temp2 = image2 - mean2
    vec_temp2 = rearrange(vec_temp2, 'h w c -> c (h w)')
    image_kl = np.copy(image2)
    image_kl = np.dot(eigvec_removed, vec_temp2)
    
    comp_rate = compression_rate(image2, image_kl)
    
    inv_eigvec_removed = la.pinv(eigvec_removed)
    rebuilt_image = np.copy(image2)
    rebuilt_image = rearrange(np.dot(inv_eigvec_removed, image_kl) + np.reshape(mean2, (3,1)), 'c (h w) -> h w c', h = image2.shape[0])
        
    imageout = cv.cvtColor(rebuilt_image.astype('uint8'), cv.COLOR_RGB2BGR)
    imageout = np.clip(imageout,0,255)
    imageout= imageout.astype('uint8')
    
    mse = eqm(image2, imageout)
    p_rate = psnr(mse)
    s_rate, _ = structural_similarity(image2.astype('uint8'), imageout.astype('uint8'), full=True, channel_axis=2)
    return (imageout, comp_rate, p_rate, s_rate)


def compression_rate(original, compressed):
    """Calculate the compression rate.

    Args:
        original (np.ndarray): original image
        compressed (np.ndarray): compressed image

    Returns:
        float: compression rate
    """
    _, original_buffer = cv.imencode('.png', original)
    byte_original = original_buffer.tobytes()
    _, compressed_buffer = cv.imencode('.png', compressed)
    byte_compressed = compressed_buffer.tobytes()
    return len(byte_compressed)/len(byte_original)


def quantization(num: [int], nb_bits: [int]):
    """Quantize the pixel to a given number of bits.

    Args:
        num (int): The number to quantize
        nb_bits (int): The number of bits to quantized to the given number

    Returns:
        [int]: Returns the step for the correct number of bits
    """
    for idx, channel in enumerate(num):
        step = (MAX_RGB + 1)/(2**nb_bits[idx])
        num[idx] = int(round(channel / step)*step)
    return num


def eqm(original: np.ndarray, reconstructed: np.ndarray):
    """Calculate the mean quadratic error between 2 images.

    Args:
        original (np.ndarray): _description_
        reconstructed (np.ndarray): _description_
  
    Returns:
        float: The mean quadratic error between original and reconstructed
    """
    assert len(original.shape) == 3, 'the parameters are not images of the correct dimensions: (x, w ,3)'
    assert original.shape[2] == 3, 'the kl transform uses color model with 3 channels such as RGB and YUV'
    assert original.shape == reconstructed.shape, 'The images are not of the same dimension'
    nb_pixels = (len(original) * original.shape[1])
    se = np.square(np.subtract(original, reconstructed))
    return np.sum(se)/nb_pixels


def psnr(mse: float):
    """Calculates the psnr of a quadratic error between 2 images.

    Args:
        eqm (float): mean quadratic error between 2 images

    Returns:
        float: psnr
    """
    assert mse > 0, 'invalid MSE between original and reconstructed'
    return 10*np.log10(MAX_RGB**2/mse)


if __name__ == '__main__':

    path_to_folder = './results_mix'

    if os.path.exists(path_to_folder):
        shutil.rmtree(path_to_folder)

    os.mkdir(path_to_folder)

    with open('./results_mix/stats.txt', 'w') as statistics:

        statistics.write("These are the stats for the mix 4:4:4 quantification in RGB basis:\n")

        for image1 in DIR_LIST:
            for image2 in DIR_LIST:
                kl_image, comp_rate, p_rate, s_rate = kl_transform(image1,image2)
                
                cv.imwrite('./results_mix/{image_name}'.format(
                    image_name=image1+image2,
                ), kl_image)

                statistics.write('{image_name} PSNR: {psnr}\n'.format(image_name=image1+image2, psnr=p_rate))
                statistics.write('{image_name} SSIM: {ssim} \n'.format(image_name=image1+image2, ssim=s_rate))
                statistics.write('{image_name} Compression rate: {compression_rate}\n\n'.format(image_name=image1+image2, compression_rate=comp_rate))

