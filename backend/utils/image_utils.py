import cv2
import numpy as np
from scipy.fftpack import dct, idct

def embed_visible_watermark(image_path, watermark_path, output_path='output_image.png', wm_size=(150, 150)):
    
    image = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)

   
    wm_width, wm_height = wm_size
    watermark = cv2.resize(watermark, (wm_width, wm_height))

    
    if watermark.shape[2] == 4:
        alpha_channel = watermark[:, :, 3] / 255.0  
        watermark = watermark[:, :, :3]  
    else:
        alpha_channel = np.ones((wm_height, wm_width))  

    # Xác định vị trí chèn watermark (góc dưới bên phải)
    y_offset = image.shape[0] - wm_height - 10
    x_offset = image.shape[1] - wm_width - 10

    # Nhúng watermark bằng alpha blending
    for c in range(3):
        image[y_offset:y_offset+wm_height, x_offset:x_offset+wm_width, c] = (
            (1 - alpha_channel) * image[y_offset:y_offset+wm_height, x_offset:x_offset+wm_width, c] +
            alpha_channel * watermark[:, :, c]
        )

    
    cv2.imwrite(output_path, image)
    return output_path


import pywt

def embed_dwt_watermark(image_path, watermark_path, output_path, alpha=0.1, scale=0.25):
    image = cv2.imread(image_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    wm_h = int(image.shape[0] * scale)
    wm_w = int(image.shape[1] * scale)
    watermark_resized = cv2.resize(watermark, (wm_w, wm_h))
    watermark_norm = (np.float32(watermark_resized) - 127.5) / 127.5

    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image_ycrcb)
    y = np.float32(y)
    coeffs = pywt.dwt2(y, 'haar')
    LL, (LH, HL, HH) = coeffs

    HL_embed = HL.copy()
    HL_embed[:wm_h, :wm_w] += alpha * watermark_norm

    y_embed = pywt.idwt2((LL, (LH, HL_embed, HH)), 'haar')
    y_embed = np.clip(y_embed, 0, 255).astype(np.uint8)
    result = cv2.merge((y_embed, cr, cb))
    result = cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, result)
    return output_path, (wm_h, wm_w)