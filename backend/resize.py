import cv2

# Đọc ảnh watermark
watermark = cv2.imread("watermark.png", cv2.IMREAD_UNCHANGED)

# Kích thước mới (ví dụ: giảm còn 30% kích thước gốc)
scale_percent = 30  # Thay đổi tỷ lệ theo nhu cầu
width = int(watermark.shape[1] * scale_percent / 100)
height = int(watermark.shape[0] * scale_percent / 100)
new_size = (width, height)

# Resize watermark
watermark_resized = cv2.resize(watermark, new_size, interpolation=cv2.INTER_AREA)

# Lưu lại ảnh watermark mới
cv2.imwrite("watermark_resized.png", watermark_resized)

print("✅ Watermark đã được nén thành watermark_resized.png")
