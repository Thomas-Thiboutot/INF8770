import numpy as np
import matplotlib.pyplot as py
from numpy import linalg as LA
import sys
from einops import rearrange

def kl_transform(image_name: str):
    imagelue = py.imread("./data/" + image_name + ".png")
    image=imagelue.astype('double')
    RGB_mean = np.mean(image, axis=(0,1), keepdims=True)
    covRGB = np.zeros((3,3), dtype = "double")
    vecTemp = image - RGB_mean
    vecTemp_rearranged = rearrange(vecTemp, 'h w c -> c (h w)')
    vecProdTemp = np.dot(vecTemp_rearranged,np.transpose(vecTemp_rearranged))
    covRGB = np.add(covRGB,vecProdTemp)
    nbPixels = len(image)*len(image[0])  
    covRGB = covRGB / nbPixels
    eigval, eigvec = LA.eig(covRGB)
    eigvec = np.transpose(eigvec)
    eigvec_removed = np.copy(eigvec)
    axe_to_remove = np.argmin(LA.norm(eigvec, axis=0))
    eigvec_removed[axe_to_remove,:] = [0.0,0.0,0.0]
    d = np.subtract(vecTemp,RGB_mean)
    imageKL = np.dot(eigvec_removed,rearrange(d, 'h w c -> c (h w)'))
    imageKL_r = rearrange(imageKL, 'c (h w) -> h w c', w=len(image[1]))
    print(imageKL_r[10][10][:])
    
 
 
 
if __name__ == '__main__':
    kl_transform(sys.argv[1])