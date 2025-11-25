from google.colab import files
import cv2
import numpy as np
from PIL import Image
from IPython.display import display

print("Upload an image file:")
uploaded = files.upload()

# Get uploaded filename
filename = list(uploaded.keys())[0]
print("✔ Uploaded:", filename)

# Load image using OpenCV (BGR format)
img_bgr = cv2.imread(filename)

# Convert BGR → RGB for correct color interpretation
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Convert to float for computations
R, G, B = cv2.split(img_rgb.astype("float32"))

# Desaturation grayscale:
# Gray = (max(R,G,B) + min(R,G,B)) / 2
maxRGB = np.maximum(np.maximum(R, G), B)
minRGB = np.minimum(np.minimum(R, G), B)

gray = (maxRGB + minRGB) / 2.0

# Convert to uint8
gray_u8 = np.clip(gray, 0, 255).astype("uint8")

# Convert to PIL for display
gray_img = Image.fromarray(gray_u8)
color_img = Image.fromarray(img_rgb)

print("\n Original image:")
display(color_img)

print("\n Grayscale (Desaturation) image:")
display(gray_img)

# Save to disk
output_name = "desaturated_gray.png"
gray_img.save(output_name)
print("\n Saved as:", output_name)
