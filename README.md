# Python Facematch

Flask API that compares two face images and returns a similarity score (`selfie_image_score`).

## Setup

```bash
# macOS: cmake is required to build dlib
brew install cmake

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# logging file used by the app
touch app.log
```

## Run

```bash
source venv/bin/activate
python script.py
```

Server listens on `http://0.0.0.0:8080`.

## API

### `POST /api/compare_faces`

Send **JSON raw** (`Content-Type: application/json`). Image values must be **base64** strings (not multipart file upload, not remote URLs).

```json
{
  "source_url": "<base64 image>",
  "target_url": "<base64 image>",
  "document_no": "optional",
  "datetime": "optional"
}
```

`data:image/...;base64,...` prefixes are supported.

**Aliases also accepted:**
- source: `source_url`, `source_image`, `source`, `sourceUrl`, `image1`, `selfie`
- target: `target_url`, `target_image`, `target`, `targetUrl`, `image2`, `document`

**Supported image formats (inside base64):** JPEG/JPG, PNG, WEBP, AVIF, GIF, BMP, TIFF, HEIC/HEIF

### Example (curl)

```bash
curl -X POST http://127.0.0.1:8080/api/compare_faces \
  -H "Content-Type: application/json" \
  -d '{"source_url":"'"$(base64 -i selfie.jpg)"'","target_url":"'"$(base64 -i id.png)"'"}'
```

### Success response

```json
{
  "selfie_image_score": 87.5
}
```

## Disclaimer

This API does **not** store or save submitted images. Images are processed in memory for face comparison only and are discarded after the request completes. Only request metadata (e.g. optional `document_no` / `datetime`) and errors may be written to `app.log`.

## Notes

- Use Postman/Insomnia with Body → **raw** → **JSON**, not form-data.
- `setuptools` is pinned below 81 because `face_recognition_models` still needs `pkg_resources`.
