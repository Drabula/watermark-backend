import cv2
import numpy as np

def dct_visible_watermark(image_path, watermark_path, alpha=0.3):
    img = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)

    h_img, w_img, _ = img.shape
    h_wm, w_wm, _ = watermark.shape

    # Định vị watermark ở góc dưới bên phải
    x_offset = w_img - w_wm - 10
    y_offset = h_img - h_wm - 10

    roi = img[y_offset:y_offset+h_wm, x_offset:x_offset+w_wm]

    # Trộn watermark vào ROI
    blended = cv2.addWeighted(roi, 1-alpha, watermark, alpha, 0)
    img[y_offset:y_offset+h_wm, x_offset:x_offset+w_wm] = blended

    cv2.imwrite("dct_visible.jpg", img)
    print("DCT Visible Watermarking hoàn thành. Ảnh lưu: dct_visible.jpg")

# Chạy thử nghiệm
dct_visible_watermark("C:/Users/PC/Documents/backend/compare/input.jpg", "C:/Users/PC/Documents/backend/compare/watermark.png")
