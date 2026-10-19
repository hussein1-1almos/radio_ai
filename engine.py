"""Detector engine: CPU-only, lazy-loaded, one inference at a time. No FastAPI/DICOM imports here.
Boxes are [x_min, y_min, x_max, y_max] normalized 0-1 relative to the image analysed."""
from __future__ import annotations
import os
THREADS = int(os.getenv("THREADS", max(1, (os.cpu_count() or 2) // 2)))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, str(THREADS))  # must be set before numpy/onnxruntime/torch load
import gc, hashlib, importlib.metadata, importlib.util, io, json, math, re, threading, time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
MAX_LOADED = int(os.getenv("MAX_LOADED_MODELS", "2"))  # detectors kept in RAM (LRU)
MAX_BOX_AREA = float(os.getenv("MAX_BOX_AREA", "0.85"))  # reject generic, near-whole-image boxes
REG = json.loads(Path(os.getenv("REGISTRY", BASE / "registry.json")).read_text("utf-8"))
WATCH = {"Atelectasis", "Effusion", "Consolidation", "Lung Opacity", "Pneumothorax", "Pneumonia", "Edema",
         "Mass", "Nodule", "Cardiomegaly"}
BASAL = {"Atelectasis", "Effusion", "Consolidation", "Lung Opacity"}
_cache: OrderedDict = OrderedDict()
_ver: dict = {}
_run_lock = threading.Lock()
_llm = None
_llm_used = 0.0


def P(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else BASE / q


def valid_box(b):
    """Clip to the image and reject malformed, degenerate or near-whole-image boxes. -> [x0,y0,x1,y1] | None"""
    try:
        v = [float(x) for x in b]
        if len(v) != 4 or not all(math.isfinite(x) for x in v):
            return None
    except (TypeError, ValueError):
        return None
    x0, y0, x1, y1 = (min(max(x, 0.0), 1.0) for x in v)
    if x1 - x0 < 0.002 or y1 - y0 < 0.002 or (x1 - x0) * (y1 - y0) > MAX_BOX_AREA:
        return None
    return [round(x, 4) for x in (x0, y0, x1, y1)]


# ---------- detectors: fn(cfg, model, img: PIL 'L', arr: float | None) -> (dets, notes) ----------
# det = {"label", "score", "box": [x0,y0,x1,y1] | None (suspected, not localized), "hints": [boxes] (optional)}
def _nms(b, sc, thr):
    order, keep = sc.argsort()[::-1], []
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        iw = np.clip(np.minimum(b[i, 2], b[rest, 2]) - np.maximum(b[i, 0], b[rest, 0]), 0, None)
        ih = np.clip(np.minimum(b[i, 3], b[rest, 3]) - np.maximum(b[i, 1], b[rest, 1]), 0, None)
        inter = iw * ih
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) < thr]
    return keep


def decode_yolo(out, cfg, scale, w, h):
    """Ultralytics (v8/v11) ONNX output (1, 4+nc, N) or (1, N, 4+nc) -> dets."""
    p = out[0] if out.ndim == 3 else out
    if p.shape[0] == 4 + len(cfg["names"]):  # channels-first -> (N, 4+nc)
        p = p.T
    sc, cl = p[:, 4:].max(1), p[:, 4:].argmax(1)
    k = sc >= cfg.get("conf", 0.25)
    p, sc, cl = p[k], sc[k], cl[k]
    b = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1) / scale
    names, dets = cfg["names"], []
    for c in np.unique(cl):
        if c >= len(names) or names[c] not in cfg.get("keep", names):
            continue
        idx = np.nonzero(cl == c)[0]
        for i in _nms(b[idx], sc[idx], cfg.get("iou", 0.45)):
            x0, y0, x1, y1 = b[idx][i]
            dets.append({"label": names[c], "score": float(sc[idx][i]), "box": [x0 / w, y0 / h, x1 / w, y1 / h]})
    return dets


def _yolo(cfg, sess, img, arr):
    size, (w, h) = cfg.get("imgsz", 640), img.size
    s = size / max(w, h)
    im = img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:im.height, :im.width] = np.asarray(im)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
    return decode_yolo(sess.run(None, {sess.get_inputs()[0].name: x})[0], cfg, s, w, h), []


def _xrv(cfg, m, img, arr):
    """Classifier only: findings are 'suspected'. Lung-region hints are NOT lesion localization."""
    import torch, torchxrayvision as xrv
    cls, sg = m
    a = xrv.datasets.normalize(np.asarray(img, np.float32), 255)[None]
    with torch.no_grad():
        p = cls(torch.from_numpy(xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(a)))[None])[0].numpy()
        probs = dict(zip(cls.pathologies, map(float, p)))
        hit = sorted(((k, v) for k, v in probs.items() if k in cfg.get("watch", WATCH) and v >= cfg.get("thresh", 0.5)),
                     key=lambda kv: -kv[1])
        lungs = (sg(torch.from_numpy(xrv.datasets.XRayResizer(512)(a))[None])[0].sigmoid().numpy() > 0.5)[4:6] if hit else []
    hints = []
    for mk in lungs:
        ys, xs = np.nonzero(mk)
        if len(ys) >= 500:
            y0, y1 = ys.min(), ys.max() + 1
            hints.append([xs.min() / 512, (y1 - (y1 - y0) / 3) / 512, (xs.max() + 1) / 512, y1 / 512])
    dets = [{"label": k, "score": v, "box": None, "hints": []} for k, v in hit]
    first = next((d for d in dets if d["label"] in BASAL), None)
    if first:
        first["hints"] = hints  # attach once, to the strongest basal finding
    top5 = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
    return dets, ["Chest classifier estimates (uncalibrated): " + ", ".join(f"{k} {v:.2f}" for k, v in top5) + "."]


def _wmh(cfg, fn, img, arr):
    if arr is None:
        raise ValueError("raw MR pixels required")
    from scipy import ndimage as ndi
    out = np.asarray(fn(arr))
    if out.shape != arr.shape:
        raise ValueError(f"segmenter returned {out.shape}, expected {arr.shape}")
    lab, n = ndi.label(out > 0.5)
    h, w = out.shape
    idx = range(1, n + 1)
    return [{"label": cfg.get("label", "Lesion"), "score": float(sc),
             "box": [s[1].start / w, s[0].start / h, s[1].stop / w, s[0].stop / h]}
            for s, a, sc in zip(ndi.find_objects(lab), ndi.sum(out > 0.5, lab, idx), ndi.mean(out, lab, idx))
            if a >= cfg.get("min_px", 4)], []


_RUN = {"yolo": _yolo, "xrv": _xrv, "wmh_seg": _wmh}
_PKG = {"yolo": "onnxruntime", "xrv": "torchxrayvision", "wmh_seg": "wmh_seg"}


def _load(cfg):
    if cfg["kind"] == "yolo":
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = THREADS, 1
        return ort.InferenceSession(str(P(cfg["weights"])), o, providers=["CPUExecutionProvider"])
    if cfg["kind"] == "xrv":
        import torch, torchxrayvision as xrv
        torch.set_num_threads(THREADS)
        return xrv.models.DenseNet(weights="densenet121-res224-all").eval(), xrv.baseline_models.chestx_det.PSPNet().eval()
    import torch
    from wmh_seg import wmh_seg
    torch.set_num_threads(THREADS)
    return wmh_seg


def _get(name, cfg):
    if name in _cache:
        _cache.move_to_end(name)
        return _cache[name]
    _cache[name] = _load(cfg)
    while len(_cache) > MAX_LOADED:
        _cache.popitem(last=False)
        gc.collect()
    return _cache[name]


def installed(cfg) -> bool:
    ok = importlib.util.find_spec(_PKG[cfg["kind"]]) is not None
    return ok and (P(cfg["weights"]).is_file() if cfg["kind"] == "yolo" else True)


def version(cfg) -> str:
    """Package version, or a short SHA-256 of the weights file."""
    try:
        if cfg["kind"] == "yolo":
            p = P(cfg["weights"])
            key = (str(p), p.stat().st_mtime_ns)
            if key not in _ver:
                _ver[key] = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            return _ver[key]
        return importlib.metadata.version(_PKG[cfg["kind"]])
    except Exception:
        return "unknown"


# ---------- routing ----------
def resolve_site(*texts):
    names = {**{s: s for s in REG["sites"]}, **REG.get("aliases", {})}
    for t in texts:
        t = (t or "").lower()
        for k in sorted(names, key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", t):
                return names[k]
    return None


def coverage():
    return {s: [{"detector": d, "installed": installed(REG["detectors"][d]), "validated": bool(REG["detectors"][d].get("validated", False)),
                 "detects": REG["detectors"][d].get("detects", "")} for d in ds] for s, ds in REG["sites"].items()}


def _gate(cfg, meta) -> str:
    """'' if the detector applies to this image, else the reason it does not."""
    if meta["modality"] not in cfg.get("modalities", []):
        return f"does not handle modality {meta['modality']}"
    if cfg.get("series_contains", "").lower() not in str(meta.get("series_description", "")).lower():
        return f"series is not {cfg['series_contains']}"
    return ""


def analyze(img, arr, meta, site_hint=""):
    mod = meta["modality"]
    site = resolve_site(site_hint, meta.get("body_part"), meta.get("study_description"), meta.get("series_description"))
    models, dets, notes = [], [], []
    with _run_lock:  # one inference at a time keeps CPU and RAM bounded
        # No body part in the header: only detectors marked "auto" whose own gate passes (e.g. FLAIR MR) are tried.
        names = ([n for n, c in REG["detectors"].items() if c.get("auto") and not _gate(c, meta)]
                 if site is None else REG["sites"].get(site, []))
        for n in names:
            cfg = REG["detectors"][n]
            rec = {"name": n, "kind": cfg["kind"], "version": None, "validated": bool(cfg.get("validated", False)),
                   "detects": cfg.get("detects", ""), "status": "ran", "detail": ""}
            models.append(rec)
            if _gate(cfg, meta):
                rec.update(status="not_applicable", detail=_gate(cfg, meta))
            elif not installed(cfg):
                rec.update(status="not_installed", detail="package or weights missing")
            else:
                rec["version"] = version(cfg)
                if site is None:
                    notes.append(cfg.get("auto_note", "Body part was not stated in the DICOM header and was assumed."))
                try:
                    d, nt = _RUN[cfg["kind"]](cfg, _get(n, cfg), img, arr)
                    dets += [{**x, "model": n} for x in d]
                    notes += nt
                except Exception as exc:
                    print(f"detector {n} failed: {exc!r}")
                    rec.update(status="failed", detail=type(exc).__name__)
    return {"site": site, "modality": mod, "models": models, "dets": dets, "notes": notes}


def build_report(r, meta):
    mod, site, models = r["modality"], r["site"], r["models"]
    ran = sum(m["status"] == "ran" for m in models)
    failed = [m["name"] for m in models if m["status"] == "failed"]
    anns, suspected, hints, limits = [], [], [], list(r["notes"])
    for d in r["dets"]:
        m = next(x for x in models if x["name"] == d["model"])
        b = valid_box(d["box"]) if d.get("box") is not None else None
        if d.get("box") is not None and b is None:
            limits.append(f"A '{d['label']}' box from '{d['model']}' failed validation and was discarded.")
        if b:
            anns.append({"id": f"a{len(anns) + 1}", "label": d["label"], "box": b,
                         "box_format": "[x_min, y_min, x_max, y_max], normalized 0-1",
                         "score": round(d["score"], 3), "score_calibrated": False, "source_model": d["model"],
                         "model_version": m["version"], "validated": m["validated"],
                         "image_ref": {"sop_instance_uid": meta.get("sop_instance_uid"),
                                       "series_instance_uid": meta.get("series_instance_uid"), "slice": None}})
        else:
            suspected.append(d)
        for h in d.get("hints") or []:
            if (hb := valid_box(h)):
                hints.append({"label": d["label"], "box": hb, "source_model": d["model"],
                              "note": "lung-region hint; not lesion localization"})
    if anns:
        status = "localized"
        summary = (f"Automated detector(s) localized {len(anns)} suspected abnormality(ies): "
                   + "; ".join(f"{a['label']} (uncalibrated score {a['score']:.2f})" for a in anns) + ".")
    elif failed:
        status = "analysis_failed"
        summary = ("Automated analysis could not be completed (" + ", ".join(failed)
                   + "). The absence of detections must not be read as a negative result.")
    elif suspected:
        status = "localization_unavailable"
        summary = ("Automated analysis suspects: " + "; ".join(f"{d['label']} (uncalibrated score {d['score']:.2f})" for d in suspected)
                   + ". Reliable localization is not available, so no region is marked.")
    elif ran:
        status = "none_detected"
        summary = "The supported detector(s) flagged no abnormality. This does not establish that the examination is normal."
    else:
        status = "unsupported"
        why = ("the body part could not be determined" if not site else
               f"no detector is configured for '{site}'" if not models else
               "no configured detector could run (" + ", ".join(f"{m['name']}: {m['status']}" for m in models) + ")")
        summary = f"{mod} examination: automated abnormality detection was not performed because {why}."
    if failed and status == "localized":
        limits.append("Detector(s) failed and were skipped: " + ", ".join(failed) + ".")
    limits += ["Single image; limited assessment, not a complete-study review.",
               "Detector output is not validated on this institution's population unless stated in 'models'.",
               "Urgent-finding detection is not performed; possible_urgent_finding is not assessed.",
               "No clinical history or prior studies were available."]
    desc = (meta.get("series_description") or "").strip()
    return {"examination": meta.get("study_description") or f"{mod} {site or '(body part undetermined)'}",
            "technique": f"{mod}" + (f", series '{desc}'" if desc else "") + f", {meta.get('columns')}x{meta.get('rows')} pixels. "
                         "Analysis covers this single image only.",
            "findings": {"summary": summary, "limitations": limits, "needs_additional_images": True,
                         "possible_urgent_finding": None,
                         "bounding_box": [[round(y0 * 1000, 1), round(x0 * 1000, 1), round(y1 * 1000, 1), round(x1 * 1000, 1)]
                                          for x0, y0, x1, y1 in (a["box"] for a in anns)],  # legacy client: [ymin,xmin,ymax,xmax] 0-1000
                         "flagged": [f"{a['label']} ({a['score']:.2f})" for a in anns]},
            "impression": [summary], "uncertainty": "Moderate" if status == "localized" else "High",
            "annotations": anns, "region_hints": hints, "annotation_status": status,
            "annotated_image_available": False, "annotated_image_url": None, "models": models,
            "recommendations": [], "sources": [], "review_required": True}


def annotate(img, anns):
    """Separate annotated copy; the input image is not modified."""
    base = img.convert("RGBA")
    w, h = base.size
    over = Image.new("RGBA", base.size)
    d = ImageDraw.Draw(over)
    gold = (255, 190, 0, 255)
    for a in anns:
        x0, y0, x1, y1 = a["box"]
        r = [x0 * w, y0 * h, x1 * w, y1 * h]
        d.rectangle(r, fill=(255, 200, 0, 45), outline=gold, width=max(2, w // 250))
        ty = max(0, r[1] - 16)
        d.rectangle([r[0], ty, r[0] + d.textlength(a["label"]) + 8, ty + 15], fill=(0, 0, 0, 190))
        d.text((r[0] + 4, ty + 2), a["label"], fill=gold)
    d.text((8, h - 16), "AI preliminary output - radiologist review required", fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, over).convert("RGB")


# ---------- optional narrative: text-only, grounded on detector results (never sees pixels) ----------
_BAD = re.compile(r"\b(normal|unremarkable|no acute|coordinates?)\b|\d\.\d{2,}\s*,\s*\d", re.I)


def _prompt(summary):
    return ("Rewrite this automated-analysis result as a short preliminary radiology draft for a radiologist. "
            f"Result: {summary} Reply with JSON: 'summary' (2-3 sentences) and 'impression' (list of strings). "
            "Use only the information in the result. Do not add findings, measurements, history or recommendations, "
            "do not describe anything as normal, and do not state coordinates.")


def narrate(report):
    """-> (summary, impression, model_label) or None. LLM=0|local|api."""
    global _llm, _llm_used
    mode = {"1": "local"}.get(os.getenv("LLM", "0"), os.getenv("LLM", "0"))
    if mode not in ("local", "api"):
        return None
    prompt = _prompt(report["findings"]["summary"])
    if mode == "api":  # text only: no pixels or patient identifiers leave the server
        import requests
        key = os.getenv("REPORT_API_KEY")
        if not key:
            raise RuntimeError("REPORT_API_KEY not set")
        model = os.getenv("REPORT_API_MODEL", "gpt-6-astra")
        r = requests.post(os.getenv("REPORT_API_URL", "https://api.openai.com/v1/chat/completions"),
                          headers={"Authorization": f"Bearer {key}"}, timeout=float(os.getenv("REPORT_API_TIMEOUT", "60")),
                          json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                "response_format": {"type": "json_object"}})
        r.raise_for_status()
        content, label = r.json()["choices"][0]["message"]["content"], f"api:{model}"
    else:
        if _llm is None:
            from llama_cpp import Llama
            _llm = Llama(model_path=str(P(os.getenv("MODEL_PATH", "models/medgemma-4b-it-Q4_K_M.gguf"))),
                         n_ctx=int(os.getenv("N_CTX", "2048")), n_threads=THREADS, n_batch=256, n_gpu_layers=0, verbose=False)
        schema = {"type": "object", "properties": {"summary": {"type": "string"}, "impression": {"type": "array", "items": {"type": "string"}}},
                  "required": ["summary", "impression"]}
        out = _llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300,
                                          response_format={"type": "json_object", "schema": schema})
        _llm_used = time.monotonic()
        content, label = out["choices"][0]["message"]["content"], "local:" + Path(os.getenv("MODEL_PATH", "medgemma")).name
    try:
        j = json.loads(content)
        s, imp = str(j["summary"]).strip(), [str(x).strip() for x in j["impression"]]
    except Exception:
        return None
    if not s or _BAD.search(s + " " + " ".join(imp)):
        return None  # reject text that claims normality or invents coordinates; keep the template
    return s, imp, label


def unload_idle(seconds: float) -> None:
    global _llm
    if _llm is not None and time.monotonic() - _llm_used > seconds:
        _llm = None
        gc.collect()