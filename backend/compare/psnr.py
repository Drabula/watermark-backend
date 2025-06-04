import cv2
import numpy as np

def calculate_psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def calculate_sr(original_wm, extracted_wm):

    if original_wm.shape != extracted_wm.shape:
        raise ValueError("❌ Hai ảnh watermark không cùng kích thước")

    original_bin = (original_wm > 127).astype(np.uint8)
    extracted_bin = (extracted_wm > 127).astype(np.uint8)

    same = np.sum(original_bin == extracted_bin)
    total = original_bin.size
    sr = same / total
    return sr

original_img = cv2.imread("C:/Users/PC/Documents/backend/compare/original_image.png", cv2.IMREAD_GRAYSCALE)
watermarked_img = cv2.imread("C:/Users/PC/Documents/backend/compare/watermarked_image.png", cv2.IMREAD_GRAYSCALE)


psnr = calculate_psnr(original_img, watermarked_img)
print(f"📊 PSNR: {psnr:.2f} dB")

original_wm = cv2.imread("C:/Users/PC/Documents/backend/compare/original_watermark.png", cv2.IMREAD_GRAYSCALE)
extracted_wm = cv2.imread("C:/Users/PC/Documents/backend/compare/extracted_watermark.png", cv2.IMREAD_GRAYSCALE)

extracted_wm = cv2.resize(extracted_wm, (original_wm.shape[1], original_wm.shape[0]))

sr = calculate_sr(original_wm, extracted_wm)
print(f"📊 Similarity Ratio (SR): {sr * 100:.2f}%")
