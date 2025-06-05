import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def calculate_sr(original_wm, extracted_wm):
    if original_wm.shape != extracted_wm.shape:
        extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))

    original_bin = (original_wm > 127).astype(np.uint8)
    extracted_bin = (extracted_wm > 127).astype(np.uint8)

    same = np.sum(original_bin == extracted_bin)
    total = original_bin.size
    sr = same / total
    return sr

# ====== Đường dẫn ảnh ======
original_path = "compare/original_image.png"
watermarked_path = "compare/watermarked_image.png"
original_wm_path = "compare/original_watermark.png"
extracted_wm_path = "compare/extracted_watermark.png"

# ====== Đọc ảnh ======
original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
watermarked_img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
original_wm = cv2.imread(original_wm_path, cv2.IMREAD_GRAYSCALE)
extracted_wm = cv2.imread(extracted_wm_path, cv2.IMREAD_GRAYSCALE)

# ====== Tính chỉ số ======
psnr = calculate_psnr(original_img, watermarked_img)
sr = calculate_sr(original_wm, extracted_wm)

# ====== Vẽ biểu đồ ======
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# PSNR
axs[0].bar(['PSNR'], [psnr], color='skyblue')
axs[0].set_title('📊 PSNR (dB)')
axs[0].set_ylim(0, 60)
axs[0].text(0, psnr + 1, f"{psnr:.2f}", ha='center')

# Similarity Ratio
axs[1].bar(['Similarity Ratio'], [sr * 100], color='lightgreen')
axs[1].set_title('📊 Similarity Ratio (%)')
axs[1].set_ylim(0, 100)
axs[1].text(0, sr * 100 + 2, f"{sr * 100:.2f}%", ha='center')

plt.tight_layout()
plt.show()
