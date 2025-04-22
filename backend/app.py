from flask import Flask, request, jsonify, send_file
import os
import sys
from utils.image_utils import embed_visible_watermark, embed_invisible_watermark
from utils.video_utils import embed_watermark_in_video
from utils.extract_utils import extract_invisible_watermark
import tempfile
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
UPLOAD_FOLDER = 'temp'
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
@app.route('/embed_invisible_watermark', methods=['POST'])
def embed_invisible_watermark_api():
    try:
        # Kiểm tra tệp được gửi lên
        if 'file' not in request.files:
            return jsonify({"error": "No 'file' part in request"}), 400
        file = request.files['file']
        
        # Kiểm tra watermark
        if 'watermark' not in request.files:
            return jsonify({"error": "No 'watermark' part in request"}), 400
        watermark = request.files['watermark']

        # Kiểm tra loại tệp
        file_type = request.form.get('type', 'image')
        if file_type != 'image':
            return jsonify({"error": f"Invalid file type: {file_type}. Expected 'image'."}), 400

        # Lưu file đã tải lên
        file_path = os.path.join(UPLOAD_FOLDER, 'input.png')  # Lưu ảnh gốc với tên cố định
        file.save(file_path)

        watermark_path = os.path.join(UPLOAD_FOLDER, 'watermark.png')  # Lưu watermark với tên cố định
        watermark.save(watermark_path)

        # Gán watermark ẩn vào ảnh
        output_path = os.path.join(UPLOAD_FOLDER, 'output.png')  # Lưu ảnh đã gán watermark với tên cố định
        embed_invisible_watermark(file_path, watermark_path, output_path)

        # Trả về ảnh đã gán watermark
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(f"🔥 Lỗi Flask: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/extract_invisible_watermark', methods=['POST'])
def extract_invisible_watermark_api():
    try:
        # Kiểm tra tệp được gửi lên
        if 'file' not in request.files:
            return jsonify({"error": "No 'file' part in request"}), 400
        file = request.files['file']

        # Kiểm tra loại tệp
        file_type = request.form.get('type', 'image')
        if file_type != 'image':
            return jsonify({"error": f"Invalid file type: {file_type}. Expected 'image'."}), 400

        # Kiểm tra file gốc
        original = request.files.get('original')
        if original is None:
            return jsonify({"error": "Original image required for extraction"}), 400
        
        # Lưu file đã tải lên
        file_path = os.path.join(UPLOAD_FOLDER, 'input.png')  # Lưu ảnh nhúng watermark với tên cố định
        file.save(file_path)

        original_path = os.path.join(UPLOAD_FOLDER, 'original.png')  # Lưu ảnh gốc với tên cố định
        original.save(original_path)

        # Trích xuất watermark (dùng hàm trích xuất watermark của bạn)
        wm = extract_invisible_watermark(file_path, original_path)

        # Tạo tệp tạm thời để lưu watermark đã trích xuất
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(temp_file.name, wm)
        
        # Trả về tệp watermark đã trích xuất
        return send_file(temp_file.name, as_attachment=True)

    except Exception as e:
        print(f"🔥 Lỗi Flask extract: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
