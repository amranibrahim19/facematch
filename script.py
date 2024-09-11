from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(filename='app.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_image_from_base64(base64_data):
    try:
        # Remove any data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        image.verify()  # Verify the image integrity
        image = Image.open(BytesIO(image_data))  # Re-open image after verification
        return face_recognition.load_image_file(BytesIO(image_data))
    except Exception as e:
        logging.error(f"Error loading image from base64: {str(e)}")
        raise ValueError(f"Error loading image: {str(e)}")

def calculate_similarity(encoding1, encoding2):
    try:
        distance = np.linalg.norm(encoding1 - encoding2)
        threshold = 1.5  # Set your own threshold value here
        percentage = max(0, min(100 - (distance / threshold * 100), 100))
        return percentage
    except Exception as e:
        logging.error(f"Error calculating similarity: {str(e)}")
        raise

@app.route('/api/compare_faces', methods=['POST'])
def compare_faces():
    data = request.json
    if 'source_url' not in data or 'target_url' not in data:
        logging.error('No base64 image data provided')
        return jsonify({'error': 'No base64 image data'}), 400

    base64_source_image = data.get('source_url')
    base64_target_image = data.get('target_url')

    if not base64_source_image or not base64_target_image:
        logging.error('No images provided in the request')
        return jsonify({'error': 'No images provided'}), 400

    try:
        # Load and encode the images
        image_of_person_1 = load_image_from_base64(base64_source_image)
        image_of_person_2 = load_image_from_base64(base64_target_image)

        encoding_of_person_1 = face_recognition.face_encodings(image_of_person_1)[0]
        encoding_of_person_2 = face_recognition.face_encodings(image_of_person_2)[0]

        # Calculate similarity percentage
        similarity_percentage = calculate_similarity(encoding_of_person_1, encoding_of_person_2)

        # Example score calculation (replace with actual logic)
        selfie_image_score = similarity_percentage

        # Extract additional data
        document_no = data.get('document_no')
        datetime_str = data.get('datetime')
        
        # Optionally process or save document_no and datetime
        logging.info(f"Document Number: {document_no}")
        logging.info(f"Datetime: {datetime_str}")

        return jsonify({'selfie_image_score': selfie_image_score})

    except Exception as e:
        logging.error(f"Error during face comparison: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run()
