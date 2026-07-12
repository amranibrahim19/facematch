from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path
import numpy as np
import base64
import logging

BASE_DIR = Path(__file__).resolve().parent

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Formats Pillow can decode here (plus HEIC/HEIF when pillow-heif is installed)
SUPPORTED_IMAGE_FORMATS = {
    'JPEG', 'JPG', 'PNG', 'WEBP', 'AVIF', 'GIF', 'BMP', 'TIFF', 'MPO', 'HEIC', 'HEIF',
}

app = Flask(__name__)
CORS(app)

logging.basicConfig(filename='app.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_image_from_base64(base64_data):
    try:
        if not isinstance(base64_data, str):
            raise ValueError('Image data must be a base64 string')

        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',', 1)[1]

        image_data = base64.b64decode(base64_data, validate=False)
        image = Image.open(BytesIO(image_data))
        image.load()

        fmt = (image.format or '').upper()
        if fmt == 'JPG':
            fmt = 'JPEG'
        if fmt and fmt not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f'Unsupported image format: {fmt}. '
                f'Accepted: {", ".join(sorted(SUPPORTED_IMAGE_FORMATS))}'
            )

        # Normalize palette/alpha/CMYK/etc. to RGB for face_recognition
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return np.asarray(image)
    except UnidentifiedImageError as e:
        logging.error(f"Unrecognized image format: {str(e)}")
        raise ValueError(
            'Unrecognized image. Accepted base64 formats: '
            'JPEG/JPG, PNG, WEBP, AVIF, GIF, BMP, TIFF, HEIC/HEIF'
        ) from e
    except Exception as e:
        logging.error(f"Error loading image from base64: {str(e)}")
        raise ValueError(f"Error loading image: {str(e)}") from e

def calculate_similarity(encoding1, encoding2):
    try:
        distance = np.linalg.norm(encoding1 - encoding2)
        threshold = 1.5  
        percentage = max(0, min(100 - (distance / threshold * 100), 100))
        return percentage
    except Exception as e:
        logging.error(f"Error calculating similarity: {str(e)}")
        raise

def _pick_image_field(data, *keys):
    for key in keys:
        value = data.get(key)
        if value:
            return value
    return None

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/README.md')
def readme():
    return send_from_directory(BASE_DIR, 'README.md', mimetype='text/markdown')

@app.route('/api/compare_faces', methods=['POST'])
def compare_faces():
    data = request.get_json(silent=True, force=True)
    if not isinstance(data, dict):
        logging.error('Request body must be JSON')
        return jsonify({
            'error': 'Request body must be JSON',
            'example': {
                'source_url': '<base64 image>',
                'target_url': '<base64 image>',
            },
        }), 400

    base64_source_image = _pick_image_field(
        data, 'source_url', 'source_image', 'source', 'sourceUrl', 'image1', 'selfie'
    )
    base64_target_image = _pick_image_field(
        data, 'target_url', 'target_image', 'target', 'targetUrl', 'image2', 'document'
    )

    if not base64_source_image or not base64_target_image:
        logging.error('No base64 image data provided. Keys=%s', list(data.keys()))
        return jsonify({
            'error': 'No base64 image data',
            'required': ['source_url', 'target_url'],
            'received_keys': list(data.keys()),
            'hint': 'Send JSON with source_url and target_url as base64 strings '
                    '(data:image/...;base64,... also works)',
        }), 400

    try:
        image_of_person_1 = load_image_from_base64(base64_source_image)
        image_of_person_2 = load_image_from_base64(base64_target_image)

        encoding_of_person_1 = face_recognition.face_encodings(image_of_person_1)[0]
        encoding_of_person_2 = face_recognition.face_encodings(image_of_person_2)[0]

        similarity_percentage = calculate_similarity(encoding_of_person_1, encoding_of_person_2)

        selfie_image_score = similarity_percentage

        document_no = data.get('document_no')
        datetime_str = data.get('datetime')
        
        logging.info(f"Document Number: {document_no}")
        logging.info(f"Datetime: {datetime_str}")

        return jsonify({'selfie_image_score': selfie_image_score})

    except Exception as e:
        logging.error(f"Error during face comparison: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG') == '1')
