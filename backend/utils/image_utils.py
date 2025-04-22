import cv2
import numpy as np
from scipy.fftpack import dct, idct

def embed_visible_watermark(image_path, watermark_path, output_path='output_image.png'):
    
    # Đọc ảnh gốc và watermark
    image = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)

    # Resize watermark (20% kích thước ảnh)
    scale = 0.2
    wm_width = int(image.shape[1] * scale)
    wm_height = int(image.shape[0] * scale)
    watermark = cv2.resize(watermark, (wm_width, wm_height))

    # Kiểm tra watermark có kênh alpha không
    if watermark.shape[2] == 4:
        alpha_channel = watermark[:, :, 3] / 255.0  # Chuẩn hóa alpha
        watermark = watermark[:, :, :3]  # Lấy phần RGB
    else:
        alpha_channel = np.ones((wm_height, wm_width))  # Nếu không có alpha, coi như full opacity

    # Xác định vị trí chèn watermark (góc dưới bên phải)
    y_offset = image.shape[0] - wm_height - 10
    x_offset = image.shape[1] - wm_width - 10

    # Nhúng watermark với alpha blending
    for c in range(3):
        image[y_offset:y_offset+wm_height, x_offset:x_offset+wm_width, c] = \
            (1 - alpha_channel) * image[y_offset:y_offset+wm_height, x_offset:x_offset+wm_width, c] + \
            alpha_channel * watermark[:, :, c]

    # Lưu ảnh kết quả
    cv2.imwrite(output_path, image)
    return output_path


def embed_invisible_watermark(image_path, watermark_path, output_path='output_image.png', alpha=0.1, scale=0.25):
    # Đọc ảnh màu và chuyển sang YCrCb
    image_color = cv2.imread(image_path)
    if image_color is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại {image_path}")
    image_ycrcb = cv2.cvtColor(image_color, cv2.COLOR_BGR2YCrCb)
    
    # Tách các kênh Y, Cr, Cb
    y_channel, cr_channel, cb_channel = cv2.split(image_ycrcb)
    y_channel = np.float32(y_channel)

    # Đọc và resize watermark
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"Không tìm thấy watermark tại {watermark_path}")
    
    wm_h = int(y_channel.shape[0] * scale)
    wm_w = int(y_channel.shape[1] * scale)
    watermark = cv2.resize(watermark, (wm_w, wm_h))
    watermark = np.float32(watermark) / 255.0  # Chuẩn hóa

    # DCT kênh Y
    dct_y = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')

    # Nhúng watermark vào góc trên bên trái
    dct_y[:wm_h, :wm_w] += alpha * watermark

    # IDCT để tái tạo kênh Y mới
    y_channel_watermarked = idct(idct(dct_y.T, norm='ortho').T, norm='ortho')
    y_channel_watermarked = np.clip(y_channel_watermarked, 0, 255).astype(np.uint8)

    # Gộp lại với các kênh màu Cr, Cb
    merged_ycrcb = cv2.merge((y_channel_watermarked, cr_channel, cb_channel))

    # Chuyển về ảnh màu BGR
    watermarked_image = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Lưu ảnh
    cv2.imwrite(output_path, watermarked_image)
    return output_path