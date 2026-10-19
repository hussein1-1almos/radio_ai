#!/usr/bin/env python3
r"""Send one or more local DICOM files to the running MedGemma API.

Usage:
  # Linux/macOS
  export API_KEY='your-local-api-key'
  python test_dicom_api.py ./dicom_folder/*.dcm

  # Windows PowerShell
  $env:API_KEY = 'your-local-api-key'
  python .\test_dicom_api.py .\dicom_folder\scan1.dcm .\dicom_folder\scan2.dcm

Only sends files to the API URL you specify (default: localhost). It does not
upload files to any third-party service by itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the local DICOM draft-report API")
    parser.add_argument("files", nargs="+", help="Path(s) to single-frame DICOM files")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/dicom/draft-report",
                        help="Draft-report endpoint URL (default: localhost)")
    parser.add_argument("--timeout", type=float, default=600,
                        help="Per-file request timeout in seconds (default: 600)")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional directory to save JSON responses; omit to print only")
    args = parser.parse_args()

    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        parser.error("Set API_KEY in your environment to match the server's .env")

    files: list[Path] = []
    for pattern in args.files:
        path = Path(pattern)
        if path.is_file():
            files.append(path)
        else:
            print(f"SKIP: not a file: {path}", file=sys.stderr)

    if not files:
        print("No input files found.", file=sys.stderr)
        return 2

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Testing {path.name} ...", flush=True)
        started = time.monotonic()
        try:
            with path.open("rb") as stream:
                response = requests.post(
                    args.url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (path.name, stream, "application/dicom")},
                    timeout=args.timeout,
                )
            elapsed = time.monotonic() - started
            print(f"HTTP {response.status_code} in {elapsed:.1f}s")
            try:
                payload = response.json()
            except ValueError:
                payload = {"raw_response": response.text[:2000]}

            print(json.dumps(payload, indent=2, ensure_ascii=False))
            if response.ok:
                print("RESULT: PASS (API returned a report; this does not validate medical accuracy)\n")
                if args.output_dir:
                    out_path = args.output_dir / f"{path.stem}.report.json"
                    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
                    print(f"Saved response: {out_path}")
            else:
                failures += 1
                print("RESULT: FAIL (see HTTP error above)\n")
        except requests.RequestException as exc:
            failures += 1
            print(f"RESULT: FAIL ({type(exc).__name__}: {exc})\n", file=sys.stderr)

    print(f"Summary: {len(files) - failures}/{len(files)} requests succeeded.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
