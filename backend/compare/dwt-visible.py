import cv2
import numpy as np
import pywt

def dwt_visible_watermark(image_path, watermark_path, alpha=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    wm_resized = cv2.resize(watermark, (LL.shape[1], LL.shape[0]))

    # Nhúng watermark rõ ràng
    LL_watermarked = (1 - alpha) * LL + alpha * wm_resized

    # Khôi phục ảnh bằng IDWT
    coeffs_watermarked = (LL_watermarked, (LH, HL, HH))
    img_watermarked = pywt.idwt2(coeffs_watermarked, 'haar')
    img_watermarked = np.clip(img_watermarked, 0, 255).astype(np.uint8)

    cv2.imwrite("dwt_visible.jpg", img_watermarked)
    print("DWT Visible Watermarking hoàn thành. Ảnh lưu: dwt_visible.jpg")

# Chạy thử nghiệm
dwt_visible_watermark("C:/Users/PC/Documents/backend/compare/input.jpg", "C:/Users/PC/Documents/backend/compare/watermark.png")
