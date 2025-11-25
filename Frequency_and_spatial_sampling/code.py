import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('document.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sampling fractions
factors = [1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16]

#FREQUENCY SAMPLING 
# Fourier transform
F = np.fft.fftshift(np.fft.fft2(img_gray))
M, N = img_gray.shape
cx, cy = M//2, N//2   # center

for frac in factors:

    # radius of low-frequency region to keep
    radx = int((M/2) * frac)
    rady = int((N/2) * frac)

    # empty frequency matrix
    F_sampled = np.zeros((M, N), dtype=complex)

    # keep only central frequencies
    F_sampled[cx-radx:cx+radx, cy-rady:cy+rady] = \
        F[cx-radx:cx+radx, cy-rady:cy+rady]

    # inverse Fourier transform
    img_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(F_sampled)))

    # show
    plt.figure()
    plt.imshow(img_freq, cmap='gray')
    plt.title(f"Frequency Sampling = {frac}")
    plt.axis('off')


#SPATIAL SAMPLING
for frac in factors:

    # downsample (reduce resolution)
    small = cv2.resize(img_gray, None, fx=frac, fy=frac, interpolation=cv2.INTER_NEAREST)

    # upsample back to original size for comparison
    spatial = cv2.resize(small, (N, M), interpolation=cv2.INTER_NEAREST)

    plt.figure()
    plt.imshow(spatial, cmap='gray')
    plt.title(f"Spatial Sampling = {frac}")
    plt.axis('off')

plt.show()

