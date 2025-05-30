import subprocess
import cv2
import numpy as np
import os
import pywt

def embed_invisible_watermark_frame(frame, watermark, alpha=0.1):
    frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = np.float32(frame_ycbcr[:, :, 0])

    # Thực hiện DWT trên kênh Y
    LL, (LH, HL, HH) = pywt.dwt2(y_channel, 'haar')

    # Resize watermark cho vừa với kích thước HL
    wm_h, wm_w = HL.shape[0], HL.shape[1]
    watermark_resized = cv2.resize(watermark, (wm_w, wm_h))
    # watermark_norm = watermark_resized / 255.0
    watermark_norm = (np.float32(watermark_resized) - 127.5) / 127.5
    # Nhúng watermark vào HL band
    HL += alpha * watermark_norm

    # Khôi phục ảnh sau nhúng bằng IDWT
    y_modified = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    y_modified = np.clip(y_modified, 0, 255).astype(np.uint8)

    # Gắn lại kênh Y đã chỉnh sửa vào ảnh
    frame_ycbcr[:, :, 0] = y_modified
    result = cv2.cvtColor(frame_ycbcr, cv2.COLOR_YCrCb2BGR)
    return result

def embed_watermark_in_video(video_path, watermark_path, output_path='output_video.mp4', visible=True, position="bottom-right", alpha=0.1):
    try:
        if visible:
            position_map = {
                "top-left": "10:10",
                "top-right": "W-w-10:10",
                "bottom-left": "10:H-h-10",
                "bottom-right": "W-w-10:H-h-10",
                "center": "(W-w)/2:(H-h)/2"
            }

            if position not in position_map:
                raise ValueError(f"Vị trí không hợp lệ: {position}")

            overlay_pos = position_map[position]

            command = [
                'ffmpeg', '-y', '-i', video_path, '-i', watermark_path,
                '-filter_complex', f'[1:v]scale=100:-1[wm];[0:v][wm]overlay={overlay_pos}',
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k', '-strict', '-2',
                output_path
            ]
            subprocess.run(command, check=True)
            return output_path

        else:
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            if watermark is None:
                raise ValueError("Không thể đọc watermark.")

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
                frame_modified = embed_invisible_watermark_frame(frame, watermark, alpha)
                out.write(frame_modified)

            cap.release()
            out.release()

            # Ghép âm thanh từ video gốc
            final_output = output_path
            command = [
                'ffmpeg', '-y', '-i', "temp_video.mp4", '-i', video_path,
                '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                '-b:a', '192k', '-strict', '-2', final_output
            ]
            subprocess.run(command, check=True)

            os.remove("temp_video.mp4")
            return final_output

    except Exception as e:
        print(f"❌ Lỗi khi xử lý video: {e}")
        return None
