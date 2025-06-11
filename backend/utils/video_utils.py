import subprocess
import cv2
import numpy as np
import os
import pywt

def embed_invisible_watermark_frame(frame, watermark, alpha=0.1):
    frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = np.float32(frame_ycbcr[:, :, 0])

    LL, (LH, HL, HH) = pywt.dwt2(y_channel, 'haar')

    wm_h, wm_w = HL.shape
    watermark_resized = cv2.resize(watermark, (wm_w, wm_h))
    watermark_norm = (np.float32(watermark_resized) - 127.5) / 127.5

    HL += alpha * watermark_norm
    y_modified = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    y_modified = np.clip(y_modified, 0, 255).astype(np.uint8)

    frame_ycbcr[:, :, 0] = y_modified
    result = cv2.cvtColor(frame_ycbcr, cv2.COLOR_YCrCb2BGR)
    return result


def embed_watermark_in_video(video_path, watermark_path, output_path='output_video.mp4', visible=True, position="bottom-right", alpha=0.1):
    try:
        if visible:
            # (Giữ nguyên phần nhúng nổi bằng FFmpeg)
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
            # ✅ NHÚNG CHÌM
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

            for frame_idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # ✅ CHỈ nhúng watermark vào frame đầu
                if frame_idx == 0:
                    frame = embed_invisible_watermark_frame(frame, watermark, alpha)

                out.write(frame)

            cap.release()
            out.release()

            # ✅ Ghép lại âm thanh từ video gốc
            command = [
                'ffmpeg', '-y', '-i', "temp_video.mp4", '-i', video_path,
                '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                '-b:a', '192k', '-strict', '-2', output_path
            ]
            subprocess.run(command, check=True)
            os.remove("temp_video.mp4")
            return output_path

    except Exception as e:
        print(f"Lỗi khi xử lý video: {e}")
        return None
