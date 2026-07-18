# Python Facematch

Flask API that compares two face images and returns a similarity score (`selfie_image_score`).

Open `/` for the docs UI and a live try-it form.

## Input modes

Each of `source_url` and `target_url` accepts any of:

| Mode | Example |
|------|---------|
| Remote URL | `https://example.com/face.jpg` |
| Base64 | raw base64 or `data:image/jpeg;base64,...` |
| File upload | multipart file (JPEG, PNG, HEIC, …) |

Modes can be mixed (e.g. uploaded selfie + URL on the ID photo). Max request body: **10 MB**.

**Supported formats:** JPEG/JPG, PNG, WEBP, AVIF, GIF, BMP, TIFF, HEIC/HEIF

## Setup

```bash
# macOS: cmake is required to build dlib (Linux/Railway uses prebuilt dlib-bin)
brew install cmake

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps face-recognition==1.3.0
```

## Run

```bash
source venv/bin/activate
python script.py
```

Server listens on `http://0.0.0.0:8080` (or `$PORT` if set).

```bash
python -m unittest test_script.py
```

## Deploy (Railway)

Railpack needs an explicit start command because the app lives in `script.py` (not `app.py`/`main.py`). This repo includes:

- `Procfile` / `railpack.json` → `gunicorn --bind 0.0.0.0:$PORT script:app`
- `gunicorn` in `requirements.txt`
- Linux uses prebuilt `dlib-bin` wheels (avoids compiling dlib for 10+ minutes)
- `face-recognition` is installed with `--no-deps` so pip does not pull source `dlib`
- Runtime apt packages: `libopenblas0`, `libgomp1`, `liblapack3` (required by dlib)

Optional CORS allowlist: set `CORS_ORIGINS` (comma-separated). Default is `*`.

## API

### `POST /api/compare_faces`

#### JSON (URL or base64)

```json
{
  "source_url": "https://example.com/selfie.jpg",
  "target_url": "<base64 image>",
  "document_no": "optional",
  "datetime": "optional"
}
```

```bash
# Remote URLs
curl -X POST http://127.0.0.1:8080/api/compare_faces \
  -H "Content-Type: application/json" \
  -d '{"source_url":"https://example.com/selfie.jpg","target_url":"https://example.com/id.png"}'

# Base64
curl -X POST http://127.0.0.1:8080/api/compare_faces \
  -H "Content-Type: application/json" \
  -d '{"source_url":"'"$(base64 -i selfie.jpg)"'","target_url":"'"$(base64 -i id.png)"'"}'
```

#### Multipart upload

```bash
curl -X POST http://127.0.0.1:8080/api/compare_faces \
  -F "source_url=@selfie.jpg" \
  -F "target_url=@id.heic"
```

Form text fields can also be URLs or base64 strings instead of files.

**Field aliases**

- source: `source_url`, `source_image`, `source`, `sourceUrl`, `image1`, `selfie`
- target: `target_url`, `target_image`, `target`, `targetUrl`, `image2`, `document`

### Success response

```json
{
  "selfie_image_score": 87.5
}
```

**Score:** `100 - (face_distance / 1.0 * 100)`, clamped to 0–100. `face_recognition` typically treats distance &lt; 0.6 as a match.

### Common errors

| Status | Meaning |
|--------|---------|
| 400 | Missing images, invalid base64/URL, unsupported format, or no face found |
| 413 | Request body larger than 10 MB |
| 500 | Unexpected server error (details stay in logs) |

## Disclaimer

This API does **not** store or save submitted images. Images are processed in memory for face comparison only and are discarded after the request completes. Optional `document_no` / `datetime` and errors may be written to process logs (stdout).

## Notes

- Postman: Body → raw JSON, or form-data with file/text fields named `source_url` / `target_url`.
- `setuptools` is pinned below 81 because `face_recognition_models` still needs `pkg_resources`.
