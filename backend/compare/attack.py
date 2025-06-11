import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import os

# ==== Đường dẫn ảnh ====
original_path = "compare/original_image.png"
watermarked_path = "compare/watermarked_image.png"
original_wm_path = "compare/original_watermark.png"
extracted_wm_path = "compare/extracted_watermark.png"

# ==== Kiểm tra file tồn tại ====
for path in [original_path, watermarked_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[❌] File không tồn tại: {path}")

# ==== Hàm tính PSNR ====
def calculate_psnr(original, modified):
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# ==== Các tấn công ====
def jpeg_compression(image_path, quality=20):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[❌] Không đọc được ảnh: {image_path}")
    out_path = "compare/attack_jpeg.jpg"
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out_path

def add_noise(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"[❌] Không đọc được ảnh: {image_path}")
    noisy_img = random_noise(img, mode='s&p', amount=0.02)
    noisy_img = (255 * noisy_img).astype(np.uint8)
    out_path = "compare/attack_noise.jpg"
    cv2.imwrite(out_path, noisy_img)
    return out_path

def blur_attack(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[❌] Không đọc được ảnh: {image_path}")
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    out_path = "compare/attack_blur.jpg"
    cv2.imwrite(out_path, blurred_img)
    return out_path

def brightness_contrast(image_path, alpha=1.2, beta=30):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[❌] Không đọc được ảnh: {image_path}")
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    out_path = "compare/attack_brightness.jpg"
    cv2.imwrite(out_path, adjusted)
    return out_path

def rotation_attack(image_path, angle=10):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[❌] Không đọc được ảnh: {image_path}")
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    out_path = "compare/attack_rotation.jpg"
    cv2.imwrite(out_path, rotated_img)
    return out_path

# ==== Thực hiện tấn công trên ảnh watermarked ====
attack_functions = {
    "JPEG Compression": jpeg_compression,
    "Noise Attack": add_noise,
    "Blurring": blur_attack,
    "Brightness Change": brightness_contrast,
    "Rotation": rotation_attack
}

# ==== Chạy và hiển thị ====
results = {}
original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(14, 6))
for i, (name, func) in enumerate(attack_functions.items()):
    out_path = func(watermarked_path)
    attacked = cv2.imread(out_path)

    if attacked is None:
        continue

    psnr = calculate_psnr(original, cv2.cvtColor(attacked, cv2.COLOR_BGR2GRAY))
    results[name] = out_path

    plt.subplot(2, 3, i + 1)
    if attacked.ndim == 3:
        plt.imshow(cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(attacked, cmap='gray')
    plt.title(f"{name}\nPSNR: {psnr:.2f} dB")
    plt.axis("off")

plt.tight_layout()
plt.show()
from skimage.metrics import structural_similarity as ssim

# ==== Hàm tính SSIM ====
def calculate_ssim(original, extracted):
    return ssim(original, extracted, data_range=extracted.max() - extracted.min())

# ==== Đọc ảnh watermark gốc và trích xuất ====
original_wm = cv2.imread(original_wm_path, cv2.IMREAD_GRAYSCALE)
wm_dct = cv2.imread("compare/extracted_watermark_dct.png", cv2.IMREAD_GRAYSCALE)
wm_dwt = cv2.imread("compare/extracted_watermark_dwt.png", cv2.IMREAD_GRAYSCALE)

if original_wm is None or wm_dct is None or wm_dwt is None:
    raise FileNotFoundError("[❌] Thiếu ảnh watermark để so sánh")

# ==== Tính toán ====
psnr_dct = calculate_psnr(original_wm, wm_dct)
ssim_dct = calculate_ssim(original_wm, wm_dct)

psnr_dwt = calculate_psnr(original_wm, wm_dwt)
ssim_dwt = calculate_ssim(original_wm, wm_dwt)

# ==== Hiển thị kết quả ====
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_wm, cmap='gray')
plt.title("Watermark Gốc")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(wm_dct, cmap='gray')
plt.title(f"DCT\nPSNR: {psnr_dct:.2f} dB\nSSIM: {ssim_dct:.3f}")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(wm_dwt, cmap='gray')
plt.title(f"DWT\nPSNR: {psnr_dwt:.2f} dB\nSSIM: {ssim_dwt:.3f}")
plt.axis('off')

plt.tight_layout()
plt.show()
