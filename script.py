from __future__ import annotations

import base64
import logging
import os
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import face_recognition
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage

BASE_DIR = Path(__file__).resolve().parent

# face_recognition typically treats distance < 0.6 as a match; map 0 → 100% and this → 0%.
SIMILARITY_DISTANCE_MAX = 1.0
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
URL_TIMEOUT_SECONDS = 15

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

SUPPORTED_IMAGE_FORMATS = {
    'JPEG', 'JPG', 'PNG', 'WEBP', 'AVIF', 'GIF', 'BMP', 'TIFF', 'MPO', 'HEIC', 'HEIF',
}

SOURCE_IMAGE_KEYS = ('source_url', 'source_image', 'source', 'sourceUrl', 'image1', 'selfie')
TARGET_IMAGE_KEYS = ('target_url', 'target_image', 'target', 'targetUrl', 'image2', 'document')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

cors_origins = [origin.strip() for origin in os.environ.get('CORS_ORIGINS', '*').split(',') if origin.strip()]
CORS(app, origins=cors_origins)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError('Empty image data')

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as error:
        logging.error('Unrecognized image format: %s', error)
        raise ValueError(
            'Unrecognized image. Accepted formats: '
            'JPEG/JPG, PNG, WEBP, AVIF, GIF, BMP, TIFF, HEIC/HEIF'
        ) from error

    image_format = (image.format or '').upper()
    if image_format == 'JPG':
        image_format = 'JPEG'
    if image_format and image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f'Unsupported image format: {image_format}. '
            f'Accepted: {", ".join(sorted(SUPPORTED_IMAGE_FORMATS))}'
        )

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return np.asarray(image)


def load_image_from_base64(base64_data: object) -> np.ndarray:
    if not isinstance(base64_data, str):
        raise ValueError('Image data must be a base64 string')

    if base64_data.startswith('data:image'):
        base64_data = base64_data.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(base64_data, validate=True)
    except Exception as error:
        logging.error('Invalid base64 image data: %s', error)
        raise ValueError('Invalid base64 image data') from error

    return load_image_from_bytes(image_bytes)


def load_image_from_url(url: str) -> np.ndarray:
    if not url.startswith(('http://', 'https://')):
        raise ValueError('Image URL must start with http:// or https://')

    http_request = urllib.request.Request(
        url,
        headers={'User-Agent': 'python-facematch/1.0'},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=URL_TIMEOUT_SECONDS) as response:
            image_bytes = response.read(MAX_CONTENT_LENGTH + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        logging.error('Failed to download image from URL: %s', error)
        raise ValueError('Failed to download image from URL') from error

    if len(image_bytes) > MAX_CONTENT_LENGTH:
        raise ValueError('Downloaded image too large')

    return load_image_from_bytes(image_bytes)


def load_image_from_upload(upload: FileStorage) -> np.ndarray:
    image_bytes = upload.read()
    if not image_bytes:
        raise ValueError('Uploaded file is empty')
    return load_image_from_bytes(image_bytes)


def is_http_url(value: str) -> bool:
    return value.startswith(('http://', 'https://'))


def load_image_input(
    value: object | None,
    upload: FileStorage | None,
    label: str,
) -> np.ndarray:
    if upload is not None and upload.filename:
        return load_image_from_upload(upload)

    if isinstance(value, str) and value.strip():
        value = value.strip()
        if is_http_url(value):
            return load_image_from_url(value)
        return load_image_from_base64(value)

    raise ValueError(
        f'No {label} image provided. Use a URL, base64 string, or file upload.'
    )


def calculate_similarity(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    distance = float(face_recognition.face_distance([encoding1], encoding2)[0])
    return max(0.0, min(100.0 - (distance / SIMILARITY_DISTANCE_MAX * 100.0), 100.0))


def encode_face(image: np.ndarray, label: str) -> np.ndarray:
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError(f'No face found in {label} image')
    return encodings[0]


def _pick_image_field(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if value:
            return value
    return None


def _pick_upload(files: Any, keys: tuple[str, ...]) -> FileStorage | None:
    for key in keys:
        upload = files.get(key)
        if upload is not None and getattr(upload, 'filename', None):
            return upload
    return None


def _request_fields_and_files() -> tuple[dict[str, Any], Any] | None:
    if request.files or (request.form and not request.is_json):
        return dict(request.form), request.files

    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data, {}

    if request.form:
        return dict(request.form), request.files

    return None


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/README.md')
def readme():
    return send_from_directory(BASE_DIR, 'README.md', mimetype='text/markdown')


@app.route('/api/compare_faces', methods=['POST'])
def compare_faces():
    payload = _request_fields_and_files()
    if payload is None:
        return jsonify({
            'error': 'Send JSON or multipart form-data',
            'example': {
                'source_url': '<https URL or base64 image>',
                'target_url': '<https URL or base64 image>',
            },
            'upload_fields': ['source_url', 'target_url'],
        }), 400

    fields, files = payload
    source_value = _pick_image_field(fields, SOURCE_IMAGE_KEYS)
    target_value = _pick_image_field(fields, TARGET_IMAGE_KEYS)
    source_upload = _pick_upload(files, SOURCE_IMAGE_KEYS)
    target_upload = _pick_upload(files, TARGET_IMAGE_KEYS)

    if (not source_value and not source_upload) or (not target_value and not target_upload):
        logging.error(
            'No image data provided. Keys=%s files=%s',
            list(fields.keys()),
            list(files.keys()) if files else [],
        )
        return jsonify({
            'error': 'No image data',
            'required': ['source_url', 'target_url'],
            'received_keys': list(fields.keys()),
            'received_files': list(files.keys()) if files else [],
            'hint': 'Each image can be an http(s) URL, base64 string, or file upload '
                    '(JPEG/JPG, PNG, WEBP, AVIF, GIF, BMP, TIFF, HEIC/HEIF)',
        }), 400

    try:
        source_encoding = encode_face(
            load_image_input(source_value, source_upload, 'source'),
            'source',
        )
        target_encoding = encode_face(
            load_image_input(target_value, target_upload, 'target'),
            'target',
        )
        score = calculate_similarity(source_encoding, target_encoding)

        document_no = fields.get('document_no')
        datetime_str = fields.get('datetime')
        if document_no or datetime_str:
            logging.info('document_no=%s datetime=%s', document_no, datetime_str)

        return jsonify({'selfie_image_score': score})
    except ValueError as error:
        logging.error('Face comparison rejected: %s', error)
        return jsonify({'error': str(error)}), 400
    except Exception:
        logging.exception('Unexpected error during face comparison')
        return jsonify({'error': 'Face comparison failed'}), 500


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({'error': f'Request too large (max {MAX_CONTENT_LENGTH} bytes)'}), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG') == '1')
