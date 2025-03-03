import cv2
import numpy as np

def dct_invisible_watermark(image_path, watermark_path, alpha=0.1):
    img = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    wm_resized = cv2.resize(watermark, (y.shape[1], y.shape[0]))

    y_dct = cv2.dct(np.float32(y))

    # Nhúng watermark vào tần số trung bình
    y_dct[:wm_resized.shape[0], :wm_resized.shape[1]] += alpha * np.float32(wm_resized)

    y_idct = cv2.idct(y_dct)
    y_idct = np.clip(y_idct, 0, 255).astype(np.uint8)

    img_ycrcb = cv2.merge([y_idct, cr, cb])
    watermarked_img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite("dct_invisible.jpg", watermarked_img)
    print("DCT Invisible Watermarking hoàn thành. Ảnh lưu: dct_invisible.jpg")

# Chạy thử nghiệm
dct_invisible_watermark("C:/Users/PC/Documents/backend/compare/input.jpg", "C:/Users/PC/Documents/backend/compare/watermark.png")
