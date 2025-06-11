import cv2
import numpy as np
from scipy.fftpack import dct, idct
import pywt
import os
import tempfile

def extract_dwt_watermark(image_path, wm_shape, alpha=0.1):
    image = cv2.imread(image_path)
    y = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0])
    _, (LH, HL, HH) = pywt.dwt2(y, 'haar')

    wm_h, wm_w = wm_shape
    extracted = HL[:wm_h, :wm_w] / alpha
    extracted = np.clip(extracted, -1, 1)
    extracted = ((extracted + 1) / 2.0) * 255
    result_img = np.clip(extracted, 0, 255).astype(np.uint8)

    # Lấy thư mục temp tuyệt đối (đi lên 1 cấp từ file extract_utils.py)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /backend
    temp_dir = os.path.join(base_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    filename = f"extracted_wm_{os.path.basename(image_path)}"
    save_path = os.path.join(temp_dir, filename)
    cv2.imwrite(save_path, result_img)

    return save_path


def extract_visible_watermark(image_path, scale=0.2, margin=10):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    wm_h, wm_w = int(h * scale), int(w * scale)
    y_offset = h - wm_h - margin
    x_offset = w - wm_w - margin
    watermark_region = image[y_offset:y_offset+wm_h, x_offset:x_offset+wm_w]
    return watermark_region

def extract_invisible_watermark_from_video(video_path, watermark_shape=(64, 64), alpha=0.1):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Không thể mở video.")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Không thể đọc frame đầu tiên.")

        # ✅ Bổ sung dòng này để định nghĩa biến y
        frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = np.float32(frame_ycbcr[:, :, 0])  # Lấy kênh Y và convert về float32

        # ✅ DWT 2 chiều
        LL, (LH, HL, HH) = pywt.dwt2(y, 'haar')

        wm_h, wm_w = watermark_shape
        wm_band = HL[:wm_h, :wm_w]
        watermark_norm = wm_band / alpha
        watermark = ((watermark_norm * 127.5) + 127.5).astype(np.uint8)

        return watermark

    except Exception as e:
        print(f"Lỗi khi trích watermark: {e}")
        return None
