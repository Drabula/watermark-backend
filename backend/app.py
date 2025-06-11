from flask import Flask, request, jsonify, send_file
import os
import sys
from utils.image_utils import embed_visible_watermark, embed_dwt_watermark
from utils.video_utils import embed_watermark_in_video
from utils.extract_utils import extract_dwt_watermark, extract_invisible_watermark_from_video
import time
import uuid
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Đường dẫn tuyệt đối thư mục temp trong /backend/temp
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)


@app.route('/embed_visible_watermark', methods=['POST'])
def embed_visible_watermark_api():
    try:
        file = request.files['file']
        watermark = request.files['watermark']
        file_type = request.form.get('type', 'image')

        file_path = os.path.join(TEMP_DIR, file.filename)
        watermark_path = os.path.join(TEMP_DIR, watermark.filename)
        file.save(file_path)
        watermark.save(watermark_path)

        if file_type == 'image':
            output_path = embed_visible_watermark(file_path, watermark_path)
        elif file_type == 'video':
            output_path = embed_watermark_in_video(file_path, watermark_path, visible=True)
        else:
            return jsonify({"error": "Invalid file type"}), 400

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        print(f"🔥 Lỗi Flask: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/embed_dwt', methods=['POST'])
def api_embed_dwt():
    try:
        file = request.files['file']
        watermark = request.files['watermark']
        alpha = float(request.form.get('alpha', 0.1))
        scale = float(request.form.get('scale', 0.25))

        file_path = os.path.join(TEMP_DIR, file.filename)
        wm_path = os.path.join(TEMP_DIR, watermark.filename)
        file.save(file_path)
        watermark.save(wm_path)

        output_path = os.path.join(TEMP_DIR, 'dwt_embedded.png')
        _, shape = embed_dwt_watermark(file_path, wm_path, output_path, alpha, scale)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/embed_dwt_video', methods=['POST'])
def embed_dwt_video():
    try:
        file = request.files['file']
        watermark = request.files['watermark']
        alpha = float(request.form.get('alpha', 0.1))

        if not file or not watermark:
            return jsonify({"error": "Thiếu file hoặc watermark"}), 400

        input_path = os.path.join(TEMP_DIR, file.filename)
        wm_path = os.path.join(TEMP_DIR, watermark.filename)
        output_path = os.path.join(TEMP_DIR, f'dwt_video_{int(time.time())}.mp4')

        file.save(input_path)
        watermark.save(wm_path)

        result_path = embed_watermark_in_video(
            input_path,
            wm_path,
            output_path=output_path,
            visible=False,
            alpha=alpha
        )

        if result_path and os.path.exists(result_path):
            return send_file(result_path, as_attachment=True)
        else:
            return jsonify({"error": "❌ Không tạo được video nhúng"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/extract_dwt', methods=['POST'])
def api_extract_dwt():
    try:
        file = request.files['file']
        wm_h = int(request.form.get('wm_h'))
        wm_w = int(request.form.get('wm_w'))
        alpha = float(request.form.get('alpha', 0.1))

        file_path = os.path.join(TEMP_DIR, file.filename)
        file.save(file_path)

        extracted_path = extract_dwt_watermark(file_path, (wm_h, wm_w), alpha)

        return send_file(extracted_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/extract_dwt_video', methods=['POST'])
def extract_dwt_video():
    try:
        video_file = request.files['file']
        wm_h = int(request.form.get('wm_h'))
        wm_w = int(request.form.get('wm_w'))
        alpha = float(request.form.get('alpha', 0.1))

        temp_video_path = os.path.join(TEMP_DIR, f'{uuid.uuid4().hex}.mp4')
        os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
        video_file.save(temp_video_path)

        extracted = extract_invisible_watermark_from_video(temp_video_path, (wm_h, wm_w), alpha)
        if extracted is None:
            return jsonify({'error': '❌ Không trích được watermark. Có thể wm_h, wm_w hoặc alpha không đúng.'}), 500
        result_path = os.path.join(TEMP_DIR, f'extracted_wm_{uuid.uuid4().hex}.png')
        print(f"[DEBUG] wm_h={wm_h}, wm_w={wm_w}, alpha={alpha}")
        print(f"[DEBUG] extracted shape: {extracted.shape if extracted is not None else 'None'}")

        cv2.imwrite(result_path, extracted)

        os.remove(temp_video_path)

        return send_file(result_path, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
