import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ============================================================
# Helper: Draw text label on image
# ============================================================
def label(img, text):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (400, 40)], fill=(0, 0, 0))
    draw.text((10, 5), text, fill=(255, 255, 255), font=font)
    return img

# ============================================================
# Generate sample scene
# ============================================================
def make_scene():
    img = Image.new('RGB', (512, 384), (40, 40, 60))
    draw = ImageDraw.Draw(img)

    for y in range(384):
        shade = int(40 + (y/384) * 150)
        draw.line([(0, y), (512, y)], fill=(shade, shade//2 + 20, shade//3))

    draw.rectangle([(0, 280), (512, 384)], fill=(20, 60, 20))
    draw.ellipse([(320, 20), (480, 180)], fill=(255, 220, 80))

    return img

# ============================================================
# Compute bit-planes
# ============================================================
def compute_bitplanes(gray_arr):
    H, W = gray_arr.shape
    planes = np.zeros((8, H, W), dtype=np.uint8)
    for b in range(8):
        planes[b] = (gray_arr >> b) & 1
    return planes

# ============================================================
# Reconstruct from 3 lowest bitplanes
# ============================================================
def reconstruct_low3(planes):
    combined = (planes[0] << 0) | (planes[1] << 1) | (planes[2] << 2)
    scaled = (combined * (255/7)).astype(np.uint8)
    return scaled

# ============================================================
# Main Program
# ============================================================
output = "bitplane_outputs"
os.makedirs(output, exist_ok=True)

# Scene + exposures
scene = make_scene()
low = Image.fromarray((np.array(scene) * 0.25).clip(0,255).astype(np.uint8))
bright = Image.fromarray((np.array(scene) * 2.0).clip(0,255).astype(np.uint8))

# Convert to grayscale
scene_g = scene.convert("L")
low_g = low.convert("L")
bright_g = bright.convert("L")

# Save originals
label(scene_g, "Original (Gray)").save(f"{output}/original_gray.png")
label(low_g, "Low Light (Gray)").save(f"{output}/low_gray.png")
label(bright_g, "Bright Light (Gray)").save(f"{output}/bright_gray.png")

# Convert to arrays
A = np.array(scene_g)
L = np.array(low_g)
B = np.array(bright_g)

# Compute bitplanes
A_bits = compute_bitplanes(A)
L_bits = compute_bitplanes(L)
B_bits = compute_bitplanes(B)

# Save bit-plane images
for b in range(8):
    bp = (A_bits[b] * 255).astype(np.uint8)
    label(Image.fromarray(bp), f"Bit-plane {b}").save(f"{output}/bitplane_{b}.png")

# Reconstructions
A_re = reconstruct_low3(A_bits)
L_re = reconstruct_low3(L_bits)
B_re = reconstruct_low3(B_bits)

label(Image.fromarray(A_re), "Reconstruction (3 lowest bits)").save(f"{output}/recon_original.png")
label(Image.fromarray(L_re), "Reconstruction (3 lowest bits)").save(f"{output}/recon_low.png")
label(Image.fromarray(B_re), "Reconstruction (3 lowest bits)").save(f"{output}/recon_bright.png")

# Differences
A_diff = np.abs(A - A_re).astype(np.uint8)
L_diff = np.abs(L - L_re).astype(np.uint8)
B_diff = np.abs(B - B_re).astype(np.uint8)

label(Image.fromarray(A_diff), "Difference = Original - Recon").save(f"{output}/diff_original.png")
label(Image.fromarray(L_diff), "Difference = Low - Recon").save(f"{output}/diff_low.png")
label(Image.fromarray(B_diff), "Difference = Bright - Recon").save(f"{output}/diff_bright.png")

print("✔ All output images saved in:", output)
