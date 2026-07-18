from __future__ import annotations

import base64
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from script import (
    app,
    calculate_similarity,
    encode_face,
    is_http_url,
    load_image_from_base64,
    load_image_from_bytes,
    load_image_from_url,
    load_image_input,
)


def _png_bytes(color: tuple[int, int, int] = (200, 100, 50), size: tuple[int, int] = (64, 64)) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', size, color).save(buffer, format='PNG')
    return buffer.getvalue()


def _png_base64(color: tuple[int, int, int] = (200, 100, 50), size: tuple[int, int] = (64, 64)) -> str:
    return base64.b64encode(_png_bytes(color, size)).decode('ascii')


class LoadImageTests(unittest.TestCase):
    def test_loads_valid_png(self):
        image = load_image_from_base64(_png_base64())
        self.assertEqual(image.shape[2], 3)

    def test_loads_data_uri(self):
        image = load_image_from_base64(f'data:image/png;base64,{_png_base64()}')
        self.assertEqual(image.shape[2], 3)

    def test_loads_from_bytes(self):
        image = load_image_from_bytes(_png_bytes())
        self.assertEqual(image.shape[2], 3)

    def test_rejects_invalid_base64(self):
        with self.assertRaises(ValueError) as context:
            load_image_from_base64('not!!!valid')
        self.assertIn('Invalid base64', str(context.exception))

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            load_image_from_base64(123)

    def test_is_http_url(self):
        self.assertTrue(is_http_url('https://example.com/a.jpg'))
        self.assertFalse(is_http_url(_png_base64()))


class LoadImageInputTests(unittest.TestCase):
    def test_prefers_upload_over_value(self):
        upload = MagicMock()
        upload.filename = 'face.png'
        upload.read.return_value = _png_bytes()
        image = load_image_input('https://example.com/ignored.jpg', upload, 'source')
        self.assertEqual(image.shape[2], 3)

    @patch('script.load_image_from_url')
    def test_uses_url(self, mock_from_url):
        mock_from_url.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        load_image_input('https://cdn.example.com/face.jpg', None, 'source')
        mock_from_url.assert_called_once_with('https://cdn.example.com/face.jpg')

    def test_uses_base64(self):
        image = load_image_input(_png_base64(), None, 'source')
        self.assertEqual(image.shape[2], 3)

    def test_missing_raises(self):
        with self.assertRaises(ValueError) as context:
            load_image_input(None, None, 'source')
        self.assertIn('No source image', str(context.exception))


class UrlLoadTests(unittest.TestCase):
    @patch('script.urllib.request.urlopen')
    def test_downloads_image(self, mock_urlopen):
        response = MagicMock()
        response.read.return_value = _png_bytes()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        mock_urlopen.return_value = response

        image = load_image_from_url('https://example.com/face.png')
        self.assertEqual(image.shape[2], 3)

    def test_rejects_non_http(self):
        with self.assertRaises(ValueError):
            load_image_from_url('ftp://example.com/face.png')


class SimilarityTests(unittest.TestCase):
    @patch('script.face_recognition.face_distance', return_value=np.array([0.0]))
    def test_identical_encodings_score_100(self, _mock_distance):
        encoding = np.zeros(128)
        self.assertEqual(calculate_similarity(encoding, encoding), 100.0)

    @patch('script.face_recognition.face_distance', return_value=np.array([1.0]))
    def test_max_distance_scores_zero(self, _mock_distance):
        encoding = np.zeros(128)
        self.assertEqual(calculate_similarity(encoding, encoding), 0.0)


class EncodeFaceTests(unittest.TestCase):
    @patch('script.face_recognition.face_encodings', return_value=[])
    def test_no_face_raises(self, _mock_encodings):
        with self.assertRaises(ValueError) as context:
            encode_face(np.zeros((64, 64, 3), dtype=np.uint8), 'source')
        self.assertEqual(str(context.exception), 'No face found in source image')


class CompareFacesApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_rejects_non_json(self):
        response = self.client.post(
            '/api/compare_faces',
            data='not-json',
            content_type='text/plain',
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_rejects_missing_images(self):
        response = self.client.post('/api/compare_faces', json={'document_no': '1'})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload['error'], 'No image data')

    @patch('script.encode_face', side_effect=ValueError('No face found in source image'))
    @patch('script.load_image_input', return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    def test_no_face_returns_400(self, _mock_load, _mock_encode):
        response = self.client.post(
            '/api/compare_faces',
            json={'source_url': _png_base64(), 'target_url': _png_base64()},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['error'], 'No face found in source image')

    @patch('script.calculate_similarity', return_value=87.5)
    @patch('script.encode_face', return_value=np.zeros(128))
    @patch('script.load_image_input', return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    def test_success_base64_json(self, _mock_load, _mock_encode, _mock_score):
        response = self.client.post(
            '/api/compare_faces',
            json={'source_url': _png_base64(), 'target_url': _png_base64()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['selfie_image_score'], 87.5)

    @patch('script.calculate_similarity', return_value=91.0)
    @patch('script.encode_face', return_value=np.zeros(128))
    @patch('script.load_image_input', return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    def test_success_remote_urls(self, mock_load, _mock_encode, _mock_score):
        response = self.client.post(
            '/api/compare_faces',
            json={
                'source_url': 'https://example.com/selfie.jpg',
                'target_url': 'https://example.com/id.png',
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['selfie_image_score'], 91.0)
        self.assertEqual(mock_load.call_count, 2)

    @patch('script.calculate_similarity', return_value=80.0)
    @patch('script.encode_face', return_value=np.zeros(128))
    @patch('script.load_image_input', return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    def test_success_multipart_upload(self, mock_load, _mock_encode, _mock_score):
        response = self.client.post(
            '/api/compare_faces',
            data={
                'source_url': (BytesIO(_png_bytes()), 'selfie.jpg'),
                'target_url': (BytesIO(_png_bytes()), 'id.png'),
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['selfie_image_score'], 80.0)
        self.assertEqual(mock_load.call_count, 2)

    @patch('script.encode_face', side_effect=RuntimeError('boom'))
    @patch('script.load_image_input', return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    def test_unexpected_error_hides_internals(self, _mock_load, _mock_encode):
        response = self.client.post(
            '/api/compare_faces',
            json={'source_url': _png_base64(), 'target_url': _png_base64()},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()['error'], 'Face comparison failed')


if __name__ == '__main__':
    unittest.main()
