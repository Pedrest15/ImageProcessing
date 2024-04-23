########################## HEADER ##################################
# NAME: Pedro Lucas Castro de Andrade                              #
# YEAR/SEMESTER: 2024/1                                            #
# ASSIGNMENT 2 - Fourier Transform & Filtering in Frequency Domain #
####################################################################

import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt

def show_images(data, rows, columns, figsize=(10, 4)):
    fig, axes = plt.subplots(figsize=figsize, nrows=rows, ncols=columns, dpi=150)

    texts = {0:'Input Image',1:'Filter',
            2:'Filtered Image',3:'Reference Image'}

    for i in range(rows):
      for j in range(columns):
        ax = axes[i, j]
        idx = i*columns+j
        ax.imshow(data[idx], cmap="gray")
        curr_label = f"{texts[idx]}"
        ax.title.set_text(curr_label)
        ax.axis('off')
    plt.show()

class Filter:
    def __init__(self,img):
        self._img = img
        self._rows,self._cols = img.shape
        self._filtered_img = None        

    def ideal_low_pass(self,r:int):
        """
            Create an ideal low-pass filter

            Args:
            - r: cut off frequency
        """
        center_row = self._rows//2
        center_col = self._cols//2

        #meshgrid of frequencies
        u,v = np.meshgrid(np.arange(self._cols),np.arange(self._rows))

        #distance of each frequency from center
        distance = np.sqrt((u-center_col)**2 + (v-center_row)**2)

        self._filtered_img = np.zeros((self._rows,self._cols))
        self._filtered_img[distance<=r] = 1

    def ideal_high_pass(self,r:int):
        """
            Create an ideal high-pass filter

            Args:
            - r: cut off frequency
        """
        filter.ideal_low_pass(r)
        filter._filtered_img = 1 - filter._filtered_img

    def ideal_band_stop(self,r1:int,r0:int):
        """
            Create an ideal band-stop filter

            Args:
            - r1: high cut of frequency
            - r0: low cut of frequency
        """
        center_row = self._rows//2
        center_col = self._cols//2

        #meshgrid of frequencies
        u,v = np.meshgrid(np.arange(self._cols),np.arange(self._rows))

        #distance of each frequency from center
        distance = np.sqrt((u-center_col)**2 + (v-center_row)**2)

        self._filtered_img = np.zeros((self._rows,self._cols))
        self._filtered_img[distance>r1] = 1
        self._filtered_img[distance<r0] = 1

    def laplacian(self):
        """
            Create an laplacian high-pass filter
        """
        center_row = self._rows//2
        center_col = self._cols//2

        #meshgrid of frequencies
        u,v = np.meshgrid(np.arange(self._cols),np.arange(self._rows))

        self._filtered_img = -4*(np.pi**2)*((u-center_col)**2 + (v-center_row)**2)

    def gaussian(self,sigma1:int,sigma2:int):
        """
            Create an gaussian low-pass filter
        """
        center_row = self._rows//2
        center_col = self._cols//2

        #meshgrid of frequencies
        u,v = np.meshgrid(np.arange(self._cols),np.arange(self._rows))

        x = ((u - center_col)**2) / (2 * sigma2**2) + ((v - center_row)**2) / (2 * sigma1**2)

        self._filtered_img = np.exp(-x)

class FreqDomain:
    def __init__(self):
        pass
    
    @classmethod
    def get_freq_domain(cls,I):
        """
            Compute frequency domain of an image

            Return:
            - image in frequency domain
        """
        return np.fft.fftshift(np.fft.fft2(I))

    @classmethod
    def multiply_imgs(cls,F1,F2):
        """
            Multiply frequency domain representations
        of two images

            Return:
            - Result of operation
        """
        return np.multiply(F1,F2)

    @classmethod
    def get_spatial_domain(cls,G):
        """
            Compute spatial domain representation from
        frequency domain

            Return:
            - image in spatial domain 
        """
        return np.real(np.fft.ifft2(np.fft.ifftshift(G)))        

    @classmethod
    def normalize(cls,G):
        """
            Normalize image values to [0,255] range

            Return:
            - normalized image
        """
        return (G-np.min(G)) / (np.max(G)-np.min(G))*255

def root_mean_squared_error(G:np.array,H:np.array) -> float:
    """
        This function calculates the Root Mean Squared Error
    between two images.

        Args:
        - G: image.
        - H: reference image.

        Return:
        - Error value.
    """
    m,n = G.shape

    return np.sqrt(np.sum((G-H)**2)/(m*n))


if __name__ == '__main__':
    
    # Take inputs
    I_name:str = input().rstrip()
    H_name:str = input().rstrip()
    filter_index:int = int(input())

    # Read input image
    I = imageio.imread(I_name)
    
    # Read reference image
    H = imageio.imread(H_name)

    # use the correct filter based on filter index
    F = FreqDomain.get_freq_domain(I)
    filter = Filter(F)

    if filter_index == 0:
        r = int(input())
        filter.ideal_low_pass(r)
    elif filter_index == 1:
        r = int(input())
        filter.ideal_high_pass(r)
        
    elif filter_index == 2:
        r1 = int(input())
        r0 = int(input())
        filter.ideal_band_stop(r1,r0)
    
    elif filter_index == 3:
        filter.laplacian()
    
    elif filter_index == 4:
        sigma1 = int(input())
        sigma2 = int(input())
        filter.gaussian(sigma1,sigma2)

    G = FreqDomain.multiply_imgs(F,filter._filtered_img)
    G = FreqDomain.get_spatial_domain(G)
    G = FreqDomain.normalize(G)

    print(root_mean_squared_error(G,H))

    # compare images
    show_images([I,filter._filtered_img,G,H],2,2)
