############## HEADER ##################
# NAME: Pedro Lucas Castro de Andrade  #
# YEAR/SEMESTER: 2024/1                #
# ASSIGNMENT 3 - Morphology + Color    #
########################################

import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt

def show_images(data, rows, columns, figsize=(10, 4)):
    fig, axes = plt.subplots(figsize=figsize, nrows=rows, ncols=columns, dpi=150)

    texts = {0:'Input Image',1:'Mask',2:'Grayscale Image',3:'Colored Image',
            4:'Mixed Image',5:'Reference Image'}

    for i in range(rows):
      for j in range(columns):
        ax = axes[i, j]
        idx = i*columns+j
        ax.imshow(data[idx], cmap="gray")
        curr_label = f"{texts[idx]}"
        ax.title.set_text(curr_label)
        ax.axis('off')
    plt.show()

def convert_into_grayscale(I:np.array) -> np.array:
    """
        Turn image into grayscale if input image has rgb,
    otherwise just return the image.
    
        Parameters:
        - I: target image.

        Returns:
        - I in grayscale.
    """
    if len(I.shape) > 2:
        coefs = np.array([0.2989, 0.5870, 0.1140])
        return np.sum(I * coefs, axis=2).astype(np.uint8)
    
    return I

def thresholding(I: np.array, L:int) -> np.array:
    """
        Set the image's pixels to 0 for values below the 
    threshold and to 1 for values above.

        Parameters:
        - I: target image.
        - L: threshold value.

        Returns:
        - filtred I.
    """
    # create a new image with ones
    I_tr = np.ones(I.shape).astype(np.uint8)
    # setting to 0 the pixels below the threshold
    I_tr[np.where(I < L)] = 0
    return I_tr 

def otsu(I:np.array, max_L:int) -> np.array:
    """
        Implements Otsu's method for automatic image thresholding.

        Parameters:
        - I: target image.
        - max_L: maximum threshold value.

        Returns:
        - Thresholded image.
    """
    # number of pixels in image
    M = np.prod(I.shape)
    
    min_var = np.inf
    min_L = max_L
    hist_t,_ = np.histogram(I, bins=256, range=(0,256))
    
    for L in np.arange(1, max_L):

        I_ti = thresholding(I, L)
        
        # computing weights
        w_a = np.sum(hist_t[:L])/float(M)
        w_b = np.sum(hist_t[L:])/float(M)
        
        # computing variances
        if w_a >= 1:
            sig_a = np.var(I[np.where(I_ti == 0)])
            var = w_a*sig_a
        elif w_b >= 1:
            sig_b = np.var(I[np.where(I_ti == 1)])
            var = w_b*sig_b
        else:
            sig_a = np.var(I[np.where(I_ti == 0)])
            sig_b = np.var(I[np.where(I_ti == 1)])
            var = w_a*sig_a + w_b*sig_b

        # swap the min_L value if L is a better thershold
        if var < min_var:
            min_var = var
            min_L = L
        
    return thresholding(I, min_L)

def erosion(I:np.array,m:int,n:int,kernel:np.array) -> np.array:
    """
        Performs erosion operation on the image 
    using the specified kernel.

        Parameters:
        - I: target image.
        - m: height of I.
        - n: width of I.
        - kernel: erosion kernel.

        Returns:
        - Eroded image.
    """
    # pad the input image to handle boundary pixels
    padded_image = np.pad(I, ((1, 1), (1, 1)), mode='constant')

    # initialize an array for the eroded image
    eroded_image = np.zeros_like(I,dtype=np.float64)

    # iterate over each pixel in I
    for i in range(m):
        for j in range(n):
            # handle boundary pixels by copying them directly
            if i == 0 or i == m-1 or j == 0 or j == n-1:
                eroded_image[i, j] = I[i,j]
            else:
                # perform erosion by taking the minimum value in the region
                eroded_image[i, j] = np.min(padded_image[i:i+3, j:j+3] * kernel)

    return eroded_image

def dilation(I:np.array,m:int,n:int,kernel:np.array) -> np.array:
    """
        Performs dilation operation on the image 
    using the specified kernel.

        Parameters:
        - I: target image.
        - m: height of I.
        - n: Width of I.
        - kernel: Dilation kernel.

        Returns:
        Dilated image.
    """
    # pad the input image to handle boundary pixels
    padded_image = np.pad(I, ((1, 1), (1, 1)), mode='constant')

    # initialize an array for the dilated image
    dilated_image = np.zeros_like(I,dtype=np.float64)

    # iterate over each pixel in I
    for i in range(m):
        for j in range(n):
            # handle boundary pixels by copying them directly
            if i == 0 or i == m-1 or j == 0 or j == n-1:
                dilated_image[i, j] = I[i,j]
            else:
                # perform dilation by taking the maximum value in the region
                dilated_image[i, j] = np.max(padded_image[i:i+3, j:j+3] * kernel)

    return dilated_image

def filter_gaussian(P, Q):
    """
        Create an gaussian low-pass filter
    """
    s1 = P
    s2 = Q

    # Compute Distances
    D = np.zeros([P, Q],dtype=np.float64)
    for u in range(P):
        for v in range(Q):
            x = (u-(P/2))**2/(2*s1**2) + (v-(Q/2))**2/(2*s2**2)
            D[u, v] = np.exp(-x)
    return D

def map_value_to_color(value, min_val, max_val, colormap):
    """
        Function to map values to colors.
    """
    
    # Scale the value to the range [0, len(colormap) - 1]
    scaled_value = (value - min_val) / (max_val - min_val) * (len(colormap) - 1)
    # Determine the two closest colors in the colormap
    idx1 = int(scaled_value)
    idx2 = min(idx1 + 1, len(colormap) - 1)
    # Interpolate between the two colors based on the fractional part
    frac = scaled_value - idx1
    color = [
        (1 - frac) * colormap[idx1][0] + frac * colormap[idx2][0],
        (1 - frac) * colormap[idx1][1] + frac * colormap[idx2][1],
        (1 - frac) * colormap[idx1][2] + frac * colormap[idx2][2]
    ]
    return color

def normalize(I:np.array) -> np.array:
    """
        Normalize images value to [0,255] range.

        Parameters:
        - I: target image.

        Return:
        - normalized image.
    """
    I = np.clip(I,0, 255)*255
    return I.astype(np.uint8)

def normalize_gray_rgb(I:np.array) -> np.array :
    """
        Normalize image values and turn
    the image into a RGB image.

        Parameters:
        - I: target grayscale image.

        Return:
        - normalized RGB image.
    """
    I = I / np.max(I)
    return np.stack((I,) * 3, axis=-1,dtype=np.float64) 

def root_mean_squared_error(G:np.array,H:np.array) -> float:
    """
        This function calculates the Root Mean Squared Error
    between two images.

        Parameters:
        - G: target image.
        - H: reference image.

        Return:
        - Error value.
    """
    m,n = G.shape

    return np.sqrt(np.sum((G-H)**2)/(m*n))

def RGB_root_mean_squared_error(G:np.array,H:np.array) -> float:
    """
        This function calculates the error for each RGB color
    channel and then computes the average error across the channels.

        Parameters:
        - G: target image.
        - H: reference image.

        Return:
        - Error value.
    """
    error_R = root_mean_squared_error(G[:,:,0], H[:,:,0])
    error_G = root_mean_squared_error(G[:,:,1], H[:,:,1])
    error_B = root_mean_squared_error(G[:,:,2], H[:,:,2])
    return (error_R + error_G + error_B)/3

def make_mask(bin_img:np.array,m:int,n:int,technique_inds:str) -> np.array:
    """
        Generates a mask using the specified techniques on a binary image.

        Parameters:
        - bin_img: binary image.
        - m: Height of bin_img.
        - n: Width of bin_img.
        - technique_inds: string containing technique indices separated by whitespace.

        Returns:
        - mask generated.
    """
    # define the kernel=3 for erosion and dilation
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    # initialize the mask with the binary image
    mask = bin_img

    # iterate over each technique index provided
    for i in technique_inds.split():
        i = int(i)
        if i == 1:
            mask = erosion(mask,m,n,kernel)
        elif i == 2:
            mask = dilation(mask,m,n,kernel)
    
    return mask

def make_color_img(m:int,n:int) -> np.array:
    """
        Generates a color image based on a given distribution.

        Parameters:
        - m: height of the color image.
        - n: width of the color image.

        Returns:
        - color image as a NumPy array.
    """
    # visible spectrum
    heatmap_colors = [
        [1, 0, 1],   # Pink
        [0, 0, 1],   # Blue
        [0, 1, 0],   # Green
        [1, 1, 0],   # Yellow
        [1, 0, 0]    # Red
    ]

    # generate color distribution using a Gaussian filter
    color_distribution = filter_gaussian(m, n)
    min_val = np.min(np.array(color_distribution))
    max_val = np.max(np.array(color_distribution))

    # create an empty RGB image
    img_color = np.ones([m, n, 3],dtype=np.float64)
    
    # iterate over each mask's pixel equal to 0
    indexes = np.where(mask==0)
    for i,j in zip(indexes[0],indexes[1]):
        img_color[i, j] = map_value_to_color(color_distribution[i, j], min_val, max_val, heatmap_colors)

    return img_color

def apply_alpha(gray_img:np.array,color_img:np.array) -> np.array:
    """
        Applies alpha blending to mix a grayscale image and a color image.

        Parameters:
        - gray_img: grayscale image.
        - color_img: color image.

        Returns:
        - resulting blended image.
    """
    # define alpha value for blending
    alpha = 0.30

    # mix the grayscale image and heatmap using alpha compositing
    mixed_I = (1.0-alpha)*gray_img + alpha*color_img
    
    return normalize(mixed_I)

if __name__ == '__main__':
    
    # Take inputs
    I_name:str = input().rstrip()
    H_name:str = input().rstrip()
    technique_inds:list = input()

    # Read input image
    img = imageio.imread(I_name).astype(np.uint8)

    # Read reference image
    H = imageio.imread(H_name).astype(np.uint8)

    # Generate gray input image and take its shape
    gray_img = convert_into_grayscale(img)
    m,n = gray_img.shape

    # Apply otsu algorithm to generate binary input image
    bin_img = otsu(gray_img,255)

    # Create the mask through successive erosions and dilations on  the input image
    mask = make_mask(bin_img,m,n,technique_inds)
    
    # Generate color input image
    color_img = make_color_img(m,n)

    # Normalize gray input image and tunr into rgb
    gray_img = normalize_gray_rgb(gray_img)   
    
    # Combine gray and color input image
    mixed_img = apply_alpha(gray_img,color_img)

    # Calculates de rms error
    error = RGB_root_mean_squared_error(mixed_img,H)

    print(f"{error:.4f}")

    # compare images
    show_images([img,mask,gray_img,color_img,mixed_img,H],2,3)
