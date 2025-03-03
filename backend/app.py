from flask import Flask, request, jsonify, send_file
import os
import sys
from utils.image_utils import embed_visible_watermark, embed_invisible_watermark
from utils.video_utils import embed_watermark_in_video

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
UPLOAD_FOLDER = 'temp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# API để nhúng thủy vân hiển thị (visible watermark)
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
        print(f"🔥 Lỗi Flask: {e}")  # In lỗi ra terminal
        return jsonify({"error": str(e)}), 500


@app.route('/embed_invisible_watermark', methods=['POST'])

def embed_invisible_watermark_api():
    try:
        file = request.files['file']
        watermark = request.files['watermark']
        file_type = request.form.get('type', 'image')

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        watermark_path = os.path.join(UPLOAD_FOLDER, watermark.filename)
        file.save(file_path)
        watermark.save(watermark_path)

        if file_type == 'image':
            output_path = embed_invisible_watermark(file_path, watermark_path)
        elif file_type == 'video':
            output_path = embed_watermark_in_video(file_path, watermark_path, visible=False)
        else:
            return jsonify({"error": "Invalid file type"}), 400

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        print(f"🔥 Lỗi Flask: {e}")  # In lỗi ra terminal
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
