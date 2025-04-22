import pywt
import cv2
import numpy as np

def dwt_invisible_watermark(image_path, watermark_path, alpha=0.2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    wm_resized = cv2.resize(watermark, (LL.shape[1], LL.shape[0]))

    # Nhúng watermark vào vùng LL
    LL_watermarked = LL + alpha * wm_resized

    # Khôi phục ảnh bằng IDWT
    coeffs_watermarked = (LL_watermarked, (LH, HL, HH))
    img_watermarked = pywt.idwt2(coeffs_watermarked, 'haar')
    img_watermarked = np.clip(img_watermarked, 0, 255).astype(np.uint8)

    cv2.imwrite("dwt_invisible.jpg", img_watermarked)
    print("DWT Invisible Watermarking hoàn thành. Ảnh lưu: dwt_invisible.jpg")

# Chạy thử nghiệm
dwt_invisible_watermark("C:/Users/PC/Documents/backend/compare/input.jpg", "C:/Users/PC/Documents/backend/compare/watermark.png")
