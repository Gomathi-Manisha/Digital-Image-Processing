from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

IMAGE_PATH = r"D:/SEM 6/WhatsApp Image 2025-11-25 at 22.21.25_c37f9523.jpg"
K = 16

def compute_mse(orig, recon):
    diff = orig.astype(np.float32) - recon.astype(np.float32)
    return np.mean(diff**2)

def compute_psnr(mse, max_val=255.0):
    if mse == 0:
        return float('inf')
    return 10 * math.log10((max_val**2) / mse)

img = Image.open(IMAGE_PATH).convert("RGB")
img_arr = np.array(img)

quant_img = img.quantize(colors=K, method=0).convert("RGB")
quant_arr = np.array(quant_img)

mse = compute_mse(img_arr, quant_arr)
psnr = compute_psnr(mse)
rate = math.log2(K)

print(f"Results for K={K}")
print(f"Rate  : {rate:.2f} bits/pixel")
print(f"MSE   : {mse:.2f}")
print(f"PSNR  : {psnr:.2f} dB")

output_name = f"median_cut_output_K{K}.png"
quant_img.save(output_name)
print(f"Saved: {output_name}")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img_arr)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(quant_arr)
plt.title(f"Quantized (K={K})")
plt.axis("off")

plt.show()
