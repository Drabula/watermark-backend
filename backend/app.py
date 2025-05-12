from flask import Flask, request, jsonify, send_file
import os
import sys
from utils.image_utils import embed_visible_watermark, embed_dwt_watermark
from utils.video_utils import embed_watermark_in_video
from utils.extract_utils import extract_dwt_watermark
import tempfile
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
UPLOAD_FOLDER = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API nhúng thủy vân hiển thị
@app.route('/embed_visible_watermark', methods=['POST'])
def embed_visible_watermark_api():
    try:
        file = request.files['file']
        watermark = request.files['watermark']
        file_type = request.form.get('type', 'image')

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        watermark_path = os.path.join(UPLOAD_FOLDER, watermark.filename)
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

# API nhúng thủy vân âm
@app.route('/embed_dwt', methods=['POST'])
def api_embed_dwt():
    try:
        file = request.files['file']
        watermark = request.files['watermark']
        alpha = float(request.form.get('alpha', 0.1))
        scale = float(request.form.get('scale', 0.25))

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        wm_path = os.path.join(UPLOAD_FOLDER, watermark.filename)
        file.save(file_path)
        watermark.save(wm_path)

        output_path = os.path.join(UPLOAD_FOLDER, 'dwt_embedded.png')
        _, shape = embed_dwt_watermark(file_path, wm_path, output_path, alpha, scale)

        # ✅ TRẢ VỀ FILE NHÚNG NGAY
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/extract_dwt', methods=['POST'])
def api_extract_dwt():
    try:
        file = request.files['file']
        wm_h = int(request.form.get('wm_h'))
        wm_w = int(request.form.get('wm_w'))
        alpha = float(request.form.get('alpha', 0.1))

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        extracted = extract_dwt_watermark(file_path, (wm_h, wm_w), alpha)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, extracted)
        return send_file(temp_file.name, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
