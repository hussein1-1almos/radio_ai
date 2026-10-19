from __future__ import annotations
import asyncio, io, os, re, secrets, tempfile, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
import engine
import numpy as np
import pydicom
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydicom.pixels import apply_modality_lut, apply_voi_lut

API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(32*1024*1024)))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "64000000"))
MAX_EDGE = int(os.getenv("MAX_IMAGE_EDGE", "1024"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "4"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300"))
IDLE_UNLOAD = float(os.getenv("IDLE_UNLOAD_SECONDS", "300"))
RETENTION = float(os.getenv("RETENTION_SECONDS", "3600"))
ANN_DIR = Path(os.getenv("ANNOTATED_DIR", Path(tempfile.gettempdir()) / "radai_annotated"))
INCLUDE_PHI = os.getenv("INCLUDE_PHI", "0") == "1"
ALLOWED_MODALITIES = {"CR", "DX", "DR", "CT", "MR", "MRI", "US"}
sem, waiting = asyncio.Semaphore(1), 0

class Annotation(BaseModel):
    id: str
    label: str
    box: list[float] = Field(description="[x_min,y_min,x_max,y_max], normalized 0-1 on analysed image")
    box_format: str
    score: float | None = None
    score_calibrated: bool = False
    source_model: str
    model_version: str | None = None
    validated: bool = False
    image_ref: dict[str, Any] = Field(default_factory=dict)

class ModelRecord(BaseModel):
    name: str
    kind: str
    version: str | None = None
    validated: bool = False
    status: Literal["ran","failed","not_installed","not_applicable"]
    detects: str = ""
    detail: str = ""

class Findings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    needs_additional_images: bool = True
    possible_urgent_finding: bool | None = None
    bounding_box: list[list[float]] = Field(default_factory=list, description="Legacy [ymin,xmin,ymax,xmax] 0-1000")
    flagged: list[str] = Field(default_factory=list)

class DraftReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    examination: str = ""
    technique: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    findings: Findings = Field(default_factory=Findings)
    impression: list[str] = Field(default_factory=list)
    uncertainty: str = "High"
    annotations: list[Annotation] = Field(default_factory=list)
    region_hints: list[dict[str, Any]] = Field(default_factory=list)
    annotation_status: Literal["localized","none_detected","localization_unavailable","unsupported","analysis_failed"]
    annotated_image_available: bool = False
    annotated_image_url: str | None = None
    models: list[ModelRecord] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    review_required: bool = True

def require_api_key(authorization: str | None = Header(default=None)):
    if not API_KEY:
        raise HTTPException(503, "API authentication is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(401, "Invalid authentication credentials")

def to_uint8(a: np.ndarray):
    a = np.asarray(a, np.float32)
    ok = np.isfinite(a)
    if not ok.any(): raise ValueError("Pixel data contains no finite values")
    lo,hi = np.percentile(a[ok],[1,99])
    if hi <= lo: lo,hi = float(a[ok].min()),float(a[ok].max())
    if hi <= lo: return np.zeros(a.shape,np.uint8)
    out = (np.clip(a,lo,hi)-lo)/(hi-lo)*255
    out[~ok] = 0
    return out.astype(np.uint8)

def dicom_to_image(raw: bytes):
    try: ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    except Exception as exc: raise ValueError("Upload is not a readable DICOM file") from exc
    mod = str(getattr(ds,"Modality","")).upper().strip()
    if mod not in ALLOWED_MODALITIES: raise ValueError(f"Unsupported modality: {mod or 'missing'}")
    if "PixelData" not in ds: raise ValueError("DICOM contains no PixelData")
    if int(getattr(ds,"NumberOfFrames",1)) != 1: raise ValueError("Multiframe DICOM is not supported; submit single-frame 2D instances")
    rows,cols = int(getattr(ds,"Rows",0)),int(getattr(ds,"Columns",0))
    if rows <= 0 or cols <= 0 or rows*cols > MAX_PIXELS: raise ValueError("Image dimensions invalid or exceed configured limit")
    try: px = ds.pixel_array
    except Exception as exc: raise ValueError("Cannot decode pixels; an additional DICOM codec may be required") from exc
    photo = str(getattr(ds,"PhotometricInterpretation","")).upper()
    arr = None
    if px.ndim == 3:
        if px.shape[-1] not in (3,4) or photo == "PALETTE COLOR": raise ValueError("Unsupported color pixel layout")
        rgb = px[...,:3] if px.dtype == np.uint8 else to_uint8(px[...,:3])
        img = Image.fromarray(rgb).convert("L")
    elif px.ndim == 2:
        try: px = apply_modality_lut(px,ds)
        except Exception: px = px.astype(np.float32)
        if mod in ("MR","MRI"): arr = np.asarray(px,np.float32)
        try: g = to_uint8(np.asarray(apply_voi_lut(px,ds),np.float32))
        except Exception: g = to_uint8(np.asarray(px,np.float32))
        img = Image.fromarray(255-g if photo == "MONOCHROME1" else g, mode="L")
    else: raise ValueError("Only 2D images are supported")
    original_size = img.size
    img.thumbnail((MAX_EDGE,MAX_EDGE),Image.Resampling.BILINEAR)
    def g(tag, default="Unknown"):
        v = getattr(ds,tag,None)
        if v is None or v == "": return default
        if isinstance(v,(float,int)): return float(v) if isinstance(v,float) else int(v)
        if type(v).__name__ == "MultiValue": return [float(x) for x in v]
        return str(v).strip()
    meta = {
        "patient_name":g("PatientName"),"patient_id":g("PatientID"),"patient_birth_date":g("PatientBirthDate"),
        "patient_sex":g("PatientSex"),"patient_age":g("PatientAge"),"study_date":g("StudyDate"),
        "study_time":g("StudyTime"),"study_description":g("StudyDescription",""),
        "series_number":g("SeriesNumber",0),"series_description":g("SeriesDescription",""),
        "body_part":g("BodyPartExamined",""),"modality":mod,"accession_number":g("AccessionNumber"),
        "study_instance_uid":g("StudyInstanceUID",None),"series_instance_uid":g("SeriesInstanceUID",None),
        "sop_instance_uid":g("SOPInstanceUID",None),"instance_number":g("InstanceNumber",0),
        "manufacturer":g("Manufacturer"),"manufacturer_model_name":g("ManufacturerModelName"),
        "institution_name":g("InstitutionName"),"rows":rows,"columns":cols,
        "analysis_width":img.width,"analysis_height":img.height,
        "original_width":original_size[0],"original_height":original_size[1],
        "slice_thickness":g("SliceThickness",0.0),"pixel_spacing":g("PixelSpacing",[0.0,0.0]),
        "image_orientation_patient":g("ImageOrientationPatient",[]),"image_position_patient":g("ImagePositionPatient",[]),
        "magnetic_field_strength":g("MagneticFieldStrength",0.0),"kvp":g("KVP",0.0),
        "repetition_time":g("RepetitionTime",0.0),"echo_time":g("EchoTime",0.0),"scope":"single instance only"
    }
    if not INCLUDE_PHI:
        for k in ("patient_name","patient_id","patient_birth_date"): meta.pop(k,None)
    return img,arr,meta

def save_annotated(im):
    token = secrets.token_urlsafe(16)
    path = ANN_DIR / f"{token}.png"
    im.save(path,"PNG"); os.chmod(path,0o600)
    return token

def sweep():
    for p in ANN_DIR.glob("*.png"):
        try:
            if time.time()-p.stat().st_mtime > RETENTION: p.unlink(missing_ok=True)
        except FileNotFoundError: pass

@asynccontextmanager
async def lifespan(app):
    if hasattr(os,"nice"):
        try: os.nice(int(os.getenv("NICE","10")))
        except (OSError,ValueError): pass
    ANN_DIR.mkdir(mode=0o700,parents=True,exist_ok=True)
    async def reaper():
        while True:
            await asyncio.sleep(30)
            try: sweep(); engine.unload_idle(IDLE_UNLOAD)
            except Exception as exc: print("reaper:",repr(exc))
    task = asyncio.create_task(reaper())
    try: yield
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass

app = FastAPI(title="Radiology localization API",version="0.6.0",
 description="CPU-only MedMO localization prototype. All output is unvalidated and requires radiologist review.",lifespan=lifespan)

@app.get("/health/live")
async def live(): return {"status":"alive"}

@app.get("/health/ready",dependencies=[Depends(require_api_key)])
async def ready(): return {"status":"ready","model":engine.status(),"waiting":waiting}

@app.get("/v1/coverage",dependencies=[Depends(require_api_key)])
async def coverage(): return engine.coverage()

@app.get("/v1/annotated/{token}",dependencies=[Depends(require_api_key)])
async def annotated(token:str):
    p = ANN_DIR/f"{token}.png"
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,32}",token) or not p.is_file(): raise HTTPException(404,"Annotated image not found or expired")
    return FileResponse(p,media_type="image/png",headers={"Cache-Control":"no-store"})

@app.post("/v1/dicom/draft-report",response_model=DraftReport,dependencies=[Depends(require_api_key)])
async def create_draft_report(file:UploadFile=File(...),site:str=Form("")):
    global waiting
    raw = await file.read(MAX_UPLOAD_BYTES+1)
    await file.close()
    if len(raw)>MAX_UPLOAD_BYTES: raise HTTPException(413,"Upload exceeds configured size limit")
    if not raw: raise HTTPException(400,"Empty upload")
    try: img,arr,meta = await run_in_threadpool(dicom_to_image,raw)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    finally: del raw
    if waiting>=MAX_QUEUE: raise HTTPException(503,"Server busy, retry shortly")
    def work():
        report = engine.build_report(engine.analyze(img,arr,meta,site),meta)
        if report.get("annotations"):
            report["annotated_image_url"] = f"/v1/annotated/{save_annotated(engine.annotate(img,report['annotations']))}"
            report["annotated_image_available"] = True
        return report
    waiting += 1
    try:
        async with sem: report = await asyncio.wait_for(run_in_threadpool(work),timeout=INFERENCE_TIMEOUT)
    except asyncio.TimeoutError as exc: raise HTTPException(504,"Inference timed out; CPU worker may still finish in background") from exc
    except Exception as exc:
        print("inference error:",repr(exc)); raise HTTPException(503,"Local inference failed") from exc
    finally: waiting=max(0,waiting-1)
    report["metadata"]=meta
    report["review_required"]=True
    return report
