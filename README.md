# Direct llama-cpp-python integration

This version removes the separate `llama-server` process. FastAPI loads the model once during startup and calls `llama-cpp-python` directly.

## Compatibility warning

MedGemma 4B is based on Gemma 3. The installed `llama-cpp-python` build must expose a working `Gemma3ChatHandler` compatible with your exact GGUF and projector. Handler availability differs across package versions/builds. This app intentionally fails at startup if the configured handler is missing; it does not substitute `Gemma4ChatHandler`.

Before deployment, verify the exact handler, model and projector against the model publisher and upstream llama-cpp-python documentation. If the stable package you use lacks a compatible handler, do not rely on an unreviewed fork for a clinical system.

## Install

Create and activate a Python virtual environment, then:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

`llama-cpp-python` may need a platform-specific wheel or source build. Follow its official installation instructions and pin the tested package/build. Put model files at the paths configured in `.env` (copy `.env.example` to `.env` and replace the API key).

## Run

    uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1

Use one worker: every worker loads a separate model and consumes additional RAM. Do not use `--reload` in production.

## Request

    curl -X POST http://127.0.0.1:8000/v1/dicom/draft-report -H "Authorization: Bearer YOUR_API_KEY" -F "file=@sample.dcm"

## Scope and clinical safety

- Single-frame 2D instance only; not a complete study or series interpretation.
- Modality allowlisting is not clinical validation.
- Rendering is a baseline visualization pipeline and can remove clinically meaningful information.
- Every report is a draft and requires qualified human review.
- The urgent flag is not a validated triage signal and must never be used to rule out an emergency.
- No persistence, institutional authentication, TLS termination, audit workflow, or clinical system integration is implemented.
- Before clinical use, complete modality-specific preprocessing validation, study-level handling, representative clinical performance testing, privacy/security review, monitoring, clinician sign-off, and regulatory assessment.

## Test with real DICOM files

Use only files you are authorized to process. For initial testing, prefer de-identified DICOMs from a public teaching dataset or files de-identified under your institution's approved process. Do not send identifiable patient data to an endpoint on another machine or to an unapproved service. The API currently accepts only a single-frame 2D instance; a multi-frame DICOM will return HTTP 422. A folder containing many single-frame CT/MR slices can be tested one file at a time, but the model will not combine them into a study-level interpretation.

1. Install the requirements and configure `.env`; verify the model and projector/handler are compatible.
2. Start the API in one terminal:

       uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1

3. In another terminal, install the HTTP client if needed:

       pip install requests

4. Set `API_KEY` to the same value as in the server's `.env`, then run the included test client.

   **Windows PowerShell:**

       $env:API_KEY = "your-local-api-key"
       python .\test_dicom_api.py .\dicom_samples\sample1.dcm .\dicom_samples\sample2.dcm

   **Linux/macOS:**

       export API_KEY="your-local-api-key"
       python test_dicom_api.py ./dicom_samples/*.dcm

   To save API responses as JSON files, add `--output-dir ./test-results`. Treat saved reports as sensitive data. For a remote server, pass its URL with `--url`, and use an approved secure connection; do not expose this development server directly to the internet.

5. A successful HTTP 200 means only that the pipeline returned schema-valid JSON. Check failures such as HTTP 401 (wrong key), 422 (unsupported/invalid DICOM), 502 (invalid model JSON), or 503 (inference/model setup failure). Review logs locally and do not interpret a successful response as evidence of clinical accuracy.

### Test checklist

- [ ] Confirm `GET /health/live` returns `{"status":"alive"}`.
- [ ] Confirm a valid single-frame DICOM returns HTTP 200 and the expected JSON fields.
- [ ] Confirm a non-DICOM file and a multi-frame DICOM are rejected.
- [ ] Confirm a wrong API key returns HTTP 401.
- [ ] Test representative modalities and image types only after confirming their pixel decoding and display pipeline.
- [ ] Compare outputs with expert-reviewed reference reports; record errors and limitations. This is not a substitute for formal clinical validation.
