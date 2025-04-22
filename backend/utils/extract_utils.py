import cv2
import numpy as np
from scipy.fftpack import dct, idct

def extract_invisible_watermark(image_path, original_path, alpha=0.1):
    # Đọc ảnh màu và chuyển sang YCrCb
    image_color = cv2.imread(image_path)
    if image_color is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại {image_path}")
    image_ycrcb = cv2.cvtColor(image_color, cv2.COLOR_BGR2YCrCb)

    # Đọc ảnh gốc để tính kích thước watermark
    original_color = cv2.imread(original_path)
    if original_color is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh gốc tại {original_path}")
    original_ycrcb = cv2.cvtColor(original_color, cv2.COLOR_BGR2YCrCb)

    # Tách các kênh Y, Cr, Cb
    y_channel, cr_channel, cb_channel = cv2.split(image_ycrcb)
    y_channel = np.float32(y_channel)

    # Tính toán kích thước watermark từ ảnh gốc
    wm_h = original_ycrcb.shape[0] // 4  # Bạn có thể thay đổi tỷ lệ này nếu cần
    wm_w = original_ycrcb.shape[1] // 4  # Tỷ lệ này sẽ phụ thuộc vào kích thước watermark

    # DCT kênh Y
    dct_y = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')

    # Trích xuất watermark từ góc trên bên trái
    watermark_extracted = dct_y[:wm_h, :wm_w] / alpha

    # Chuẩn hóa lại watermark
    watermark_extracted = np.clip(watermark_extracted * 255.0, 0, 255).astype(np.uint8)

    # Trả về watermark đã trích xuất
    return watermark_extracted
