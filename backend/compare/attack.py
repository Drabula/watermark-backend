import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

def jpeg_compression(image_path, quality=20):
    img = cv2.imread(image_path)
    compressed_path = "attack_jpeg.jpg"
    cv2.imwrite(compressed_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return compressed_path

def add_noise(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_img = random_noise(img, mode='s&p', amount=0.02)  # Thêm nhiễu Salt & Pepper
    noisy_img = (255 * noisy_img).astype(np.uint8)
    cv2.imwrite("attack_noise.jpg", noisy_img)
    return "attack_noise.jpg"

def blur_attack(image_path):
    img = cv2.imread(image_path)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite("attack_blur.jpg", blurred_img)
    return "attack_blur.jpg"

def brightness_contrast(image_path, alpha=1.2, beta=30):
    img = cv2.imread(image_path)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite("attack_brightness.jpg", adjusted)
    return "attack_brightness.jpg"

def rotation_attack(image_path, angle=10):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite("attack_rotation.jpg", rotated_img)
    return "attack_rotation.jpg"

# Chạy thử nghiệm các tấn công
attacks = {
    "JPEG Compression": jpeg_compression("dct_invisible.jpg"),
    "Noise Attack": add_noise("dct_invisible.jpg"),
    "Blurring": blur_attack("dct_invisible.jpg"),
    "Brightness Change": brightness_contrast("dct_invisible.jpg"),
    "Rotation": rotation_attack("dct_invisible.jpg")
}

# Hiển thị ảnh bị tấn công
plt.figure(figsize=(12, 6))
for i, (attack, path) in enumerate(attacks.items()):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(attack)
    plt.axis("off")
plt.show()
