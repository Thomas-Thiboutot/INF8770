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


def kl_transform(image_name: str, y: int, u: int, v: int, basis: str):
    """Apply the KL transformation on images.

    Args:
        image_name (str): The name of the image to transform
        r (int): The red channel number of bits
        g (int): The green channel number of bits
        b (int): The blue channel number of bits

    Returns:
        np.ndarray: Returns the image after compression and decompression
    """
    im = cv.imread('./data/{image}'.format(image=image_name))
    image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    if basis == 'yuv':
        image = cv.cvtColor(image, cv.COLOR_RGB2YUV).astype('double')
        
    image = np.apply_along_axis(lambda channels: quantization(channels, [y, u, v]), 2, image)
    
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
    
    image_kl = np.copy(image)
    image_kl = np.dot(eigvec_removed, vec_temp_reshaped)
    
    comp_rate = compression_rate(image, image_kl)
    
    inv_eigvec_removed = la.pinv(eigvec_removed)
    rebuilt_image = np.copy(image)
    rebuilt_image = rearrange(np.dot(inv_eigvec_removed, image_kl) + np.reshape(mean, (3,1)), 'c (h w) -> h w c', h = image.shape[0])
    
    if basis == 'yuv':
        rebuilt_image = cv.cvtColor(rebuilt_image.astype('uint8'), cv.COLOR_YUV2RGB)
        
    imageout = cv.cvtColor(rebuilt_image.astype('uint8'), cv.COLOR_RGB2BGR)
    imageout = np.clip(imageout,0,255)
    imageout= imageout.astype('uint8')
    
    mse = eqm(image, imageout)
    p_rate = psnr(mse)
    s_rate, _ = structural_similarity(image.astype('uint8'), imageout.astype('uint8'), full=True, channel_axis=2)
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
    parser = argparse.ArgumentParser(
        prog='kl_transform',
        description='Applies a kl transformation to an image',
        epilog='--------------------------------',
    )

    parser.add_argument('-r', '--red', type=int)
    parser.add_argument('-g', '--green', type=int)
    parser.add_argument('-b', '--blue', type=int)
    parser.add_argument('-c', '--colormode', type=str)
    args = parser.parse_args()

    path_to_folder = './results_{basis}_{red}_{green}_{blue}'.format(
        basis=args.colormode,
        red=args.red,
        green=args.green,
        blue=args.blue,
    )

    if os.path.exists(path_to_folder):
        shutil.rmtree(path_to_folder)

    os.mkdir(path_to_folder)

    with open('./results_{basis}_{red}_{green}_{blue}/stats.txt'.format(
        basis=args.colormode,
        red=args.red,
        green=args.green,
        blue=args.blue,
    ), 'w') as statistics:

        statistics.write("These are the stats for the {red}_{green}_{blue} quantification in {basis} basis:\n".format(
            basis=args.colormode,
            red=args.red,
            green=args.green,
            blue=args.blue,
        ))

        for image in DIR_LIST:
            kl_image, comp_rate, p_rate, s_rate = kl_transform(image, args.red, args.green, args.blue, args.colormode)
            cv.imwrite('./results_{basis}_{red}_{green}_{blue}/{image_name}'.format(
                basis=args.colormode,
                image_name=image,
                red=args.red,
                green=args.green,
                blue=args.blue
            ), kl_image)

            statistics.write('{image_name} PSNR: {psnr}\n'.format(image_name=image, psnr=p_rate))
            statistics.write('{image_name} SSIM: {ssim} \n'.format(image_name=image, ssim=s_rate))
            statistics.write('{image_name} Compression rate: {compression_rate}\n\n'.format(image_name=image, compression_rate=comp_rate))

