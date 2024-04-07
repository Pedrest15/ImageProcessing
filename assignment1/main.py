################ HEADER ##########################
# NAME: Pedro Lucas Castro de Andrade            #
# YEAR/SEMESTER: 2024/1                          #
# ASSIGNMENT 1 - Superresolution and Enhancement #
##################################################

import numpy as np
import imageio.v3 as imageio
#import matplotlib.pyplot as plt

def histogram(img:np.array,num_levels:int) -> np.array:
    """
        This function calculetes the grayscale image histogram.

        Args:
        - img: target image.
        - num_levels: number of levels.

        Return:
        - histogram.
    """
    
    n,m = img.shape
    
    #the histogram has all of image number of gray levels
    hist = np.zeros(num_levels, dtype=np.int32)

    #calculates the frequence of each level
    for i in range(num_levels):
        hist[i] = np.sum(img==i)
        
    return hist

def cumulative_histogram(img:np.array,num_levels:int) -> np.array:
    """
        This function applies the cumulative histogram of an 
    image using the grayscale image histogram over the image to
    correct it.
    
        Args:
        - img: target image.
        - num_levels: number of levels.

        Return:
        - New image.
    """
    
    hist = histogram(img,num_levels)
    histC = np.zeros(num_levels,dtype=np.int32)

    # Use the histogram and then accumulate the levels
    histC[0] = hist[0]
    for i in range(1,num_levels):
        histC[i] = hist[i] + histC[i-1]

    # The new image has the same shape of the previus one
    n,m = img.shape
    img_eq = np.zeros([n,m],dtype=np.int32)

    for z in range(num_levels):
        s = ((num_levels-1)/float(n*m))*histC[z]

        img_eq[np.where(img==z)] = s

    return img_eq

def joint_cumulative_histogram(img:np.array,num_levels:int,hist:np.array) -> np.array:
    """
        This function applies the joint histogram, 
    considering the grayscale histogram of all input images,
    over the image to correct it.
    
        Args:
        - img: target image.
        - num_levels: number of levels.

        Return:
        - New image
    """
    
    histC = np.zeros(num_levels,dtype=np.int32)

    # Use the histogram and then accumulate the levels
    histC[0] = hist[0]
    for i in range(1,num_levels):
        histC[i] = hist[i] + histC[i-1]

    # The new image has the same shape of the previus one
    n,m = img.shape
    img_eq = np.zeros([n,m],dtype=np.int32)

    for z in range(num_levels):
        s = ((num_levels-1)/float(4*n*m))*histC[z]

        img_eq[np.where(img==z)] = s

    return img_eq

def gamma_correction(img:np.array,gamma:float) -> np.array:
    """
        This function applies gamma correction to an image.

        Args:
        - img: target image.
        - gamma: coeficient value.

        Return:
        - New image.
    """
    return np.floor(255*((img/255.0)**(1/gamma)))

def superresolution(low1:np.array,low2:np.array,low3:np.array,low4:np.array) -> np.array:
    """
        This function performs superresolution of an image by
    combining input images into one high-resolution one.

        Args:
        - low1;...;low4: four input images.

        Returns:
        - High-resolution image.
    """
    
    i,j=low1.shape
    # Create an empty higher resolution image super_h
    super_h = np.zeros((i*2,j*2), dtype=np.int32)

    # Compose the higher resolution image super_h
    super_h[::2, ::2] = low1
    super_h[1::2, ::2] = low2
    super_h[::2, 1::2] = low3
    super_h[1::2, 1::2] = low4

    return super_h

def root_mean_squared_error(h:np.array,super_h:np.array) -> float:
    """
        This function calculates the Root Mean Squared Error
    between two images.

        Args:
        - h: low-resolution image.
        - super_h: high-resolution image.

        Return:
        - Error value.
    """
    return np.sqrt(np.sum((h-super_h)**2)/(h.shape[0]**2))

if __name__ == '__main__':
    
    # Take inputs
    imglow:str = input().rstrip()
    imghigh:str = input().rstrip()
    F:int = int(input())
    gamma:float = float(input())

    # Read all low-resolution input images
    low_img=[None]*4
    for i in range(4):
        low_img[i] = imageio.imread(imglow+str(i)+'.png')
    
    # Read all high-resolution input images
    high_img = imageio.imread(imghigh)

    if F == 1:
        for i in range(len(low_img)):
            low_img[i] = cumulative_histogram(low_img[i],256)
    elif F == 2:
        hist=histogram(np.concatenate((low_img[0],low_img[1],low_img[2],low_img[3]),axis=1),256)
        for i in range(len(low_img)):
            low_img[i] = joint_cumulative_histogram(low_img[i],256,hist)
    elif F == 3:
        for i in range(len(low_img)):
            low_img[i] = gamma_correction(low_img[i],gamma)

    super_h = superresolution(low_img[0],low_img[1],low_img[2],low_img[3])

    print(f"{root_mean_squared_error(high_img,super_h):.4f}")