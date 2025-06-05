import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def calculate_psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_sr(original_wm, extracted_wm):
    if original_wm.shape != extracted_wm.shape:
        extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))
    original_bin = (original_wm > 127).astype(np.uint8)
    extracted_bin = (extracted_wm > 127).astype(np.uint8)
    return np.sum(original_bin == extracted_bin) / original_bin.size

def apply_transform(image, transform):
    if transform == "rotate":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif transform == "crop":
        h, w = image.shape
        return image[h//4:h*3//4, w//4:w*3//4]
    elif transform == "compress":
        # Save as JPEG (lossy), then read lại
        temp_path = "compare/compressed.jpg"
        cv2.imwrite(temp_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        return cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    return image  # no transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=str, default='none',
                        choices=['none', 'rotate', 'crop', 'compress'],
                        help='Loại biến đổi áp dụng lên ảnh đã nhúng watermark')
    args = parser.parse_args()

    # ====== Đường dẫn ======
    original_path = "compare/original_image.png"
    watermarked_path = "compare/watermarked_image.png"
    original_wm_path = "compare/original_watermark.png"
    extracted_wm_path = "compare/extracted_watermark.png"

    # ====== Đọc ảnh ======
    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    watermarked_img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
    original_wm = cv2.imread(original_wm_path, cv2.IMREAD_GRAYSCALE)
    extracted_wm = cv2.imread(extracted_wm_path, cv2.IMREAD_GRAYSCALE)

    # ====== Áp dụng biến đổi ======
    transformed_img = apply_transform(watermarked_img.copy(), args.transform)
    cv2.imwrite("compare/transformed_image.png", transformed_img)

    # ====== Tính toán ======
    psnr = calculate_psnr(original_img, transformed_img)
    sr = calculate_sr(original_wm, extracted_wm)

    # ====== Vẽ biểu đồ ======
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(['PSNR'], [psnr], color='skyblue')
    axs[0].set_title('📊 PSNR (dB)')
    axs[0].set_ylim(0, 60)
    axs[0].text(0, psnr + 1, f"{psnr:.2f}", ha='center')

    axs[1].bar(['Similarity Ratio'], [sr * 100], color='lightgreen')
    axs[1].set_title('📊 Similarity Ratio (%)')
    axs[1].set_ylim(0, 100)
    axs[1].text(0, sr * 100 + 2, f"{sr * 100:.2f}%", ha='center')

    plt.suptitle(f"📷 Biến đổi áp dụng: {args.transform}", fontsize=12)
    plt.tight_layout()
    plt.show()
