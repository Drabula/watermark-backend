import subprocess
import cv2
import numpy as np
import os

def embed_invisible_watermark_frame(frame, watermark):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(frame_yuv)

    # Chuyển kênh Y (độ sáng) sang miền DCT
    y_dct = cv2.dct(np.float32(y))

    # Resize watermark để nhúng vào miền tần số cao
    watermark_resized = cv2.resize(watermark, (y_dct.shape[1] // 4, y_dct.shape[0] // 4))
    watermark_resized = watermark_resized / 255.0

    alpha = 0.02  # Hệ số nhúng rất nhỏ để ẩn hoàn toàn watermark

    # Nhúng watermark vào vùng tần số cao
    start_x, start_y = y_dct.shape[1] // 2, y_dct.shape[0] // 2
    y_dct[start_y:start_y + watermark_resized.shape[0], start_x:start_x + watermark_resized.shape[1]] += watermark_resized * alpha

    # Chuyển lại về miền không gian bằng IDCT
    y_modified = cv2.idct(y_dct)
    y_modified = np.clip(y_modified, 0, 255).astype(np.uint8)

    # Ghép lại với kênh U, V
    frame_yuv_modified = cv2.merge((y_modified, u, v))
    frame_modified = cv2.cvtColor(frame_yuv_modified, cv2.COLOR_YUV2BGR)

    return frame_modified


def embed_watermark_in_video(video_path, watermark_path, output_path='output_video.mp4', visible=True, position="bottom-right"):
    try:
        if visible:
            # Xác định vị trí overlay watermark
            position_map = {
                "top-left": "10:10",
                "top-right": "W-w-10:10",
                "bottom-left": "10:H-h-10",
                "bottom-right": "W-w-10:H-h-10",
                "center": "(W-w)/2:(H-h)/2"
            }

            # Kiểm tra vị trí hợp lệ
            if position not in position_map:
                raise ValueError(f"Vị trí không hợp lệ: {position}")

            overlay_pos = position_map[position]

            # Nhúng watermark visible bằng FFmpeg
            command = [
                'ffmpeg', '-i', video_path, '-i', watermark_path,
                '-filter_complex', f'[1:v]scale=100:-1[wm];[0:v][wm]overlay={overlay_pos}',
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k', '-strict', '-2',
                output_path
            ]
            subprocess.run(command, check=True)
            return output_path

        else:
            # Đọc watermark ẩn
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            if watermark is None:
                raise ValueError("Không thể đọc watermark.")

            # Xử lý video với OpenCV (tránh lưu từng frame ra file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Không thể mở video.")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out = cv2.VideoWriter("temp_video.mp4", fourcc, fps, (width, height))

            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_modified = embed_invisible_watermark_frame(frame, watermark)
                out.write(frame_modified)

            cap.release()
            out.release()

            # Dùng FFmpeg để ghép âm thanh từ video gốc
            command = [
                'ffmpeg', '-i', "temp_video.mp4", '-i', video_path,
                '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                '-b:a', '192k', '-strict', '-2', output_path
            ]
            subprocess.run(command, check=True)

            os.remove("temp_video.mp4")

            return output_path

    except Exception as e:
        print(f"❌ Lỗi khi xử lý video: {e}")
        return None
