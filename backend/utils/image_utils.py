import cv2
import numpy as np

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

import cv2
import numpy as np

def embed_invisible_watermark(image_path, watermark_path, output_path='output_image.png'):
    
    # Đọc ảnh gốc và watermark
    image = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # Resize watermark để phù hợp với ảnh gốc
    watermark = cv2.resize(watermark, (image.shape[1] // 4, image.shape[0] // 4))

    # Vị trí watermark ở góc trên trái
    x_offset, y_offset = 20, 20
    h, w = watermark.shape

    # Chèn watermark vào ảnh gốc với độ mờ 10%
    alpha = 0.1  # Độ trong suốt 10%
    overlay = image.copy()
    overlay[y_offset:y_offset+h, x_offset:x_offset+w, 0] = \
        (1 - alpha) * image[y_offset:y_offset+h, x_offset:x_offset+w, 0] + alpha * watermark

    # Lưu ảnh đã xử lý
    cv2.imwrite(output_path, overlay)
    return output_path
