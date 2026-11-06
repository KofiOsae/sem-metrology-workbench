from __future__ import annotations

import base64
import asyncio
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from sem_metrology.calibration import detect_footer, detect_scale_bar
from sem_metrology.io import to_gray
from sem_metrology.lines import analyze_line_space
from sem_metrology.measurements import Measurement, measure_cross_section, measure_layer_stack
from sem_metrology.overlay import circle_overlay, cross_section_overlay, line_space_overlay, rectangular_overlay, shape_overlay
from sem_metrology.reporting import measurement_statistics
from sem_metrology.populations import infer_feature_classes
from sem_metrology.shapes import analyze_circular_outlines, analyze_rectangular_features, analyze_shapes
from sem_metrology.waveguide import analyze_waveguide, waveguide_overlay

APP_VERSION = "1.2.1-web"
ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_UPLOAD = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_ACTIVE_JOBS = 2
JOB_TTL_SECONDS = 60 * 60
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
USAGE_READY = False
USAGE_LOCK = threading.Lock()

app = FastAPI(title="SEM Metrology Workbench", version=APP_VERSION)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


def _usage_connection():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return None
    try:
        import psycopg
        return psycopg.connect(url, connect_timeout=8)
    except Exception:
        return None


def _ensure_usage_schema(conn):
    global USAGE_READY
    if USAGE_READY:
        return
    with USAGE_LOCK:
        if USAGE_READY:
            return
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS usage_events (
                event_id uuid PRIMARY KEY,
                user_hash char(64) NOT NULL,
                recipe text NOT NULL,
                app_version text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )""")
        conn.commit()
        USAGE_READY = True


def _record_usage(client_id: str, recipe: str):
    if not client_id or len(client_id) > 200:
        return False
    conn = _usage_connection()
    if conn is None:
        return False
    try:
        _ensure_usage_schema(conn)
        salt = os.getenv("USAGE_HASH_SALT", "sem-metrology-workbench")
        user_hash = hashlib.sha256(f"{salt}:{client_id}".encode()).hexdigest()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO usage_events(event_id,user_hash,recipe,app_version) VALUES (%s,%s,%s,%s)",
                        (str(uuid.uuid4()), user_hash, recipe[:80], APP_VERSION))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def _usage_summary():
    conn = _usage_connection()
    if conn is None:
        return {"persistent": False, "completed_analyses": None, "unique_users": None,
                "message": "Persistent usage counter is not configured."}
    try:
        _ensure_usage_schema(conn)
        with conn.cursor() as cur:
            cur.execute("""SELECT count(*), count(DISTINCT user_hash), min(created_at), max(created_at)
                           FROM usage_events""")
            total, users, first, last = cur.fetchone()
        return {"persistent": True, "completed_analyses": int(total), "unique_users": int(users),
                "first_use": first.isoformat() if first else None, "last_use": last.isoformat() if last else None}
    except Exception:
        return {"persistent": False, "completed_analyses": None, "unique_users": None,
                "message": "Usage counter is temporarily unavailable."}
    finally:
        conn.close()


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _roi(text: str, width: int, height: int):
    if not text: return None
    try: values = [int(round(float(v))) for v in json.loads(text)]
    except Exception as exc: raise HTTPException(400, f"Invalid ROI: {exc}")
    if len(values) != 4: raise HTTPException(400, "ROI must contain x, y, width and height")
    x, y, w, h = values; x=max(0,min(width-1,x)); y=max(0,min(height-1,y)); w=max(1,min(width-x,w)); h=max(1,min(height-y,h))
    return x, y, w, h


def _numeric_stats(rows):
    stats=[]
    if not rows:return stats
    ignored={"Object","Band","Gauge","Structure","Status","Class","Polarity","Touches border","Contact split"}
    for key in rows[0]:
        if key in ignored:continue
        values=[]
        for row in rows:
            value=row.get(key)
            if isinstance(value,(int,float,np.number)) and np.isfinite(value):values.append(float(value))
        if not values:continue
        data=np.asarray(values); mean=float(data.mean()); std=float(data.std(ddof=1)) if len(data)>1 else None
        stats.append({"Metric":key,"N":len(data),"Mean":mean,"Std":std,"Median":float(np.median(data)),"Min":float(data.min()),"Max":float(data.max()),"CV (%)":float(std/mean*100) if std is not None and mean else None})
    return stats


def _overlay_data(rgb):
    ok, encoded=cv2.imencode(".png",cv2.cvtColor(np.asarray(rgb,dtype=np.uint8),cv2.COLOR_RGB2BGR))
    if not ok:raise HTTPException(500,"Could not encode overlay")
    return "data:image/png;base64,"+base64.b64encode(encoded.tobytes()).decode("ascii")


def _scale_overlay(rgb, detection, nm_per_px=None, source=""):
    canvas=np.asarray(rgb,dtype=np.uint8).copy()
    if detection:
        a=tuple(int(v) for v in detection["start_xy"]); b=tuple(int(v) for v in detection["end_xy"])
        cv2.line(canvas,a,b,(255,0,255),max(3,round(canvas.shape[1]/450)),cv2.LINE_AA)
        label=f"SCALE DATUM {detection['pixel_length']:.1f} px"
        cv2.putText(canvas,label,(max(4,min(a[0],b[0])),max(18,min(a[1],b[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2,cv2.LINE_AA)
    if nm_per_px:
        text=f"CALIBRATION {nm_per_px:.6g} nm/px"+(f" · {source}" if source else "")
        cv2.putText(canvas,text,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,0,255),2,cv2.LINE_AA)
    return canvas


def _apply_population_classes(rows, recipe, mode):
    keys={"circles":"Diameter (µm)","rectangles":"Mean width (µm)","particles":"Feret max (µm)","grating":"Mean CD (µm)"}
    key=keys.get(recipe)
    if mode=="one" or not key or len(rows)<8 or not all(key in row and np.isfinite(row[key]) for row in rows):return None
    requested={"two":2,"three":3}.get(mode)
    model=infer_feature_classes(np.asarray([[row[key]] for row in rows],float),requested_classes=requested)
    if model.get("labels") is None:return None
    for row,label in zip(rows,model["labels"]):
        row["Size class"]=f"Class {'ABC'[int(label)]}"; row["Class confidence (%)"]=model["confidence"]*100
    return {"count":int(model["class_count"]),"confidence":float(model["confidence"]),"counts":model["counts"].tolist()}


def _recommend_model(rgb):
    gray=to_gray(rgb)
    if max(gray.shape)>1200:
        factor=1200/max(gray.shape); gray=cv2.resize(gray,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    work=cv2.GaussianBlur(gray,(3,3),0); ex=float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,1,0,ksize=3)))); ey=float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,0,1,ksize=3))))
    orientation="vertical" if ex>=ey else "horizontal"; confidence=max(ex,ey)/(ex+ey+1e-9)
    if confidence>=.72:
        candidates=[analyze_line_space(gray,1.0,p,2.0,max(gray.shape)/2,orientation)[0] for p in ("bright","dark")]
        if max(map(len,candidates))>=4:return {"recipe":"grating","orientation":orientation,"confidence":confidence}
    return None


@app.get("/")
def index(): return FileResponse(WEB / "index.html")


@app.get("/health")
def health(): return {"status":"ok","version":APP_VERSION}


@app.post("/api/preview")
async def preview(image: UploadFile = File(...)):
    """Return a browser-displayable PNG without retaining the acquisition."""
    raw = await image.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Image exceeds the 20 MB browser limit")
    try:
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(400, f"Cannot read image: {exc}")
    if rgb.shape[0] * rgb.shape[1] > MAX_PIXELS:
        raise HTTPException(413, "Image exceeds the 25 megapixel browser limit")
    footer=detect_footer(rgb); detection=detect_scale_bar(rgb,footer)
    analysis=rgb[:footer] if footer<rgb.shape[0] else rgb
    return JSONResponse(_safe({"width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                         "footer_start":footer,"scale_detection":detection,
                         "recommendation":_recommend_model(analysis),
                         "preview": _overlay_data(_scale_overlay(rgb,detection))}))


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...), recipe: str = Form(...), nm_per_px: float = Form(...),
    min_size_nm: float = Form(10), max_size_nm: float = Form(10000), sensitivity: float = Form(.82),
    polarity: str = Form("bright"), orientation: str = Form("vertical"), reject_border: bool = Form(True),
    reference_roi: str = Form(""), analysis_roi: str = Form(""), scale_line: str = Form(""),
    calibration_source: str = Form("direct"), period_x_nm: float = Form(0), period_y_nm: float = Form(0),
    layer_period_count: int = Form(0), class_mode: str = Form("auto"), classes_confirmed: bool = Form(False)
):
    raw=await image.read(MAX_UPLOAD+1)
    if len(raw)>MAX_UPLOAD:raise HTTPException(413,"Image exceeds the 20 MB browser limit")
    try: rgb=np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"),dtype=np.uint8)
    except Exception as exc:raise HTTPException(400,f"Cannot read image: {exc}")
    if rgb.shape[0]*rgb.shape[1]>MAX_PIXELS:raise HTTPException(413,"Image exceeds the 25 megapixel browser limit")
    if not np.isfinite(nm_per_px) or nm_per_px<=0:raise HTTPException(400,"Enter a positive calibrated nm/pixel value")
    footer=detect_footer(rgb); analysis=rgb[:footer].copy() if footer<rgb.shape[0] else rgb.copy(); gray=to_gray(analysis); scale=nm_per_px/1000.0
    detection=detect_scale_bar(rgb,footer)
    if scale_line:
        try:
            points=json.loads(scale_line); a,b=points
            length=float(np.hypot(float(b[0])-float(a[0]),float(b[1])-float(a[1])))
            detection={"start_xy":a,"end_xy":b,"pixel_length":length,"confidence":1.0,"style":"operator_drawn"}
        except Exception as exc:raise HTTPException(400,f"Invalid scale line: {exc}")
    ref=_roi(reference_roi,gray.shape[1],gray.shape[0]); roi=_roi(analysis_roi,gray.shape[1],gray.shape[0])
    rows=[]; measurements=[]; overlay=analysis.copy(); warnings=[]
    recommendation=_recommend_model(analysis)
    if recommendation and recipe!=recommendation["recipe"]:warnings.append(f"Model compatibility: image has a high-confidence {recommendation['orientation']} line-space signature ({recommendation['confidence']:.0%}), but {recipe} was selected.")
    min_um=min_size_nm/1000; max_um=max_size_nm/1000
    if recipe=="circles":
        rows,_,geometry=analyze_circular_outlines(gray,scale,min_um,max_um,sensitivity,reject_border,ref)
        overlay=np.asarray(circle_overlay(analysis,geometry,rows))
    elif recipe=="rectangles":
        if ref is None:raise HTTPException(400,"Draw one complete reference rectangle first")
        rows,boxes,_=analyze_rectangular_features(gray,scale,ref,sensitivity,reject_border); overlay=np.asarray(rectangular_overlay(analysis,boxes,rows))
    elif recipe in ("particles","single"):
        target=roi if recipe=="single" and roi else (0,0,gray.shape[1],gray.shape[0]); x,y,w,h=target; crop=gray[y:y+h,x:x+w]
        nominal=None; template=None
        if ref:
            rx,ry,rw,rh=ref; nominal=float(np.sqrt(rw*rh)); template=gray[ry:ry+rh,rx:rx+rw]
        min_area=max(8,int(np.pi*(min_um/scale/2)**2)); max_area=max(min_area+1,int(np.pi*(max_um/scale/2)**2))
        if polarity=="auto":
            candidates=[analyze_shapes(crop,scale,p,min_area,max_area,reject_border,nominal,sensitivity,template) for p in ("bright","dark")]
            rows,_,contours=max(candidates,key=lambda result:len(result[0]))
        else:rows,_,contours=analyze_shapes(crop,scale,polarity,min_area,max_area,reject_border,nominal,sensitivity,template)
        local=np.asarray(shape_overlay(analysis[y:y+h,x:x+w],contours,rows)); overlay[y:y+h,x:x+w]=local
        if any(bool(row.get("Contact split")) for row in rows):warnings.append("Touching particles are visible-segment estimates; hidden boundaries are not reconstructed.")
    elif recipe=="grating":
        if orientation=="auto":
            work=cv2.GaussianBlur(gray,(3,3),0); ex=float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,1,0,ksize=3)))); ey=float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,0,1,ksize=3)))); orientation="vertical" if ex>=ey else "horizontal"
        if polarity=="auto":
            candidates=[analyze_line_space(gray,scale,p,min_um,max_um,orientation) for p in ("bright","dark")]
            rows,_,bands=max(candidates,key=lambda result:len(result[0]))
        else:rows,_,bands=analyze_line_space(gray,scale,polarity,min_um,max_um,orientation)
        overlay=np.asarray(line_space_overlay(analysis,bands,orientation))
    elif recipe in ("waveguide","cross_section","layer_stack"):
        if roi is None:raise HTTPException(400,"Draw an analysis ROI around the intended structure")
        x,y,w,h=roi; crop=gray[y:y+h,x:x+w]; crop_rgb=analysis[y:y+h,x:x+w]
        if recipe=="waveguide":
            rows,diagnostics=analyze_waveguide(crop,scale); overlay[y:y+h,x:x+w]=waveguide_overlay(crop_rgb,diagnostics)
        elif recipe=="cross_section":
            measurements,diagnostics=measure_cross_section(crop,scale); overlay[y:y+h,x:x+w]=np.asarray(cross_section_overlay(crop_rgb,diagnostics))
        else:
            measurements,diagnostics=measure_layer_stack(crop,scale); diagnostics["unit_scale"]=scale*1000; diagnostics["unit"]="nm"
            if layer_period_count>0:
                period_px=w/layer_period_count; diagnostics["period_px"]=period_px; diagnostics["unit_scale"]=scale*1000; diagnostics["unit"]="nm"
                measurements.append(Measurement("Operator-guided lateral period",period_px*scale,"µm",f"ROI width divided by {layer_period_count} declared periods",1.0))
            overlay[y:y+h,x:x+w]=np.asarray(cross_section_overlay(crop_rgb,diagnostics)); warnings.append("Layer interfaces are automatic candidates and require operator review in the desktop edition.")
    else:raise HTTPException(400,"Unknown recipe")
    for axis,hint in (("X",period_x_nm),("Y",period_y_nm)):
        if hint>0:
            keys=(f"Fundamental period {axis} (µm)",f"Period {axis} (µm)")
            for row in rows:
                for key in keys:
                    if key in row:row[key]=hint/1000
                row[f"Period {axis} confidence (%)"]=100.0
    class_model=_apply_population_classes(rows,recipe,class_mode)
    if class_model and not classes_confirmed:warnings.append(f"{class_model['count']} size classes proposed ({class_model['confidence']:.0%} confidence); operator confirmation required.")
    stats=measurement_statistics(measurements) if measurements else _numeric_stats(rows)
    if not rows and not measurements:warnings.append("No credible structures were found. Review calibration, polarity, size limits, ROI and recipe.")
    display=rgb.copy(); display[:overlay.shape[0],:overlay.shape[1]]=overlay; display=_scale_overlay(display,detection,nm_per_px,calibration_source)
    return JSONResponse(_safe({"version":APP_VERSION,"recipe":recipe,"feature_count":len(rows) if rows else (1 if measurements else 0),"rows":rows,"statistics":stats,"warnings":warnings,"class_model":class_model,"overlay":_overlay_data(display),"privacy":"Image processed in memory and not intentionally retained by the application."}))


def _analysis_worker(raw: bytes, params: dict, output_path: str):
    """Run one analysis in its own OS process so cancellation stops CPU work."""
    from starlette.datastructures import UploadFile as StarletteUploadFile
    path = Path(output_path)
    temporary = path.with_suffix(".tmp")
    try:
        upload = StarletteUploadFile(file=io.BytesIO(raw), filename="browser-upload")
        response = asyncio.run(analyze(image=upload, **params))
        payload = {"ok": True, "result": json.loads(response.body)}
    except Exception as exc:
        payload = {"ok": False, "error": str(getattr(exc, "detail", exc))}
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def _cleanup_jobs():
    now = time.time()
    with JOBS_LOCK:
        expired = [key for key, job in JOBS.items() if now - job["created"] > JOB_TTL_SECONDS]
        for key in expired:
            job = JOBS.pop(key)
            process = job["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            Path(job["output"]).unlink(missing_ok=True)


@app.post("/api/jobs")
async def create_analysis_job(
    image: UploadFile = File(...), recipe: str = Form(...), nm_per_px: float = Form(...),
    min_size_nm: float = Form(10), max_size_nm: float = Form(10000), sensitivity: float = Form(.82),
    polarity: str = Form("bright"), orientation: str = Form("vertical"), reject_border: bool = Form(True),
    reference_roi: str = Form(""), analysis_roi: str = Form(""), scale_line: str = Form(""),
    calibration_source: str = Form("direct"), period_x_nm: float = Form(0), period_y_nm: float = Form(0),
    layer_period_count: int = Form(0), class_mode: str = Form("auto"), classes_confirmed: bool = Form(False),
    client_id: str = Form("")
):
    _cleanup_jobs()
    raw = await image.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Image exceeds the 20 MB browser limit")
    with JOBS_LOCK:
        active = sum(1 for job in JOBS.values() if job["process"].is_alive())
        if active >= MAX_ACTIVE_JOBS:
            raise HTTPException(429, "The analysis service is busy. Try again shortly.")
        job_id = uuid.uuid4().hex
        output = str(Path(tempfile.gettempdir()) / f"sem-metrology-{job_id}.json")
        params = dict(recipe=recipe, nm_per_px=nm_per_px, min_size_nm=min_size_nm, max_size_nm=max_size_nm,
                      sensitivity=sensitivity, polarity=polarity, orientation=orientation,
                      reject_border=reject_border, reference_roi=reference_roi, analysis_roi=analysis_roi,
                      scale_line=scale_line, calibration_source=calibration_source, period_x_nm=period_x_nm,
                      period_y_nm=period_y_nm, layer_period_count=layer_period_count,
                      class_mode=class_mode, classes_confirmed=classes_confirmed)
        process = mp.get_context("spawn").Process(target=_analysis_worker, args=(raw, params, output), daemon=False)
        process.start()
        JOBS[job_id] = {"process": process, "output": output, "created": time.time(), "recipe": recipe,
                        "client_id": client_id, "counted": False, "cancelled": False}
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def analysis_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Analysis job not found or expired")
        process = job["process"]
        if process.is_alive():
            return {"job_id": job_id, "status": "running"}
        process.join(timeout=.1)
        output = Path(job["output"])
        if job["cancelled"]:
            return {"job_id": job_id, "status": "cancelled"}
        if not output.exists():
            return {"job_id": job_id, "status": "failed", "error": "Analysis worker stopped unexpectedly"}
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            return {"job_id": job_id, "status": "failed", "error": payload.get("error", "Analysis failed")}
        should_count = not job["counted"]
        if should_count:
            job["counted"] = True
    if should_count:
        _record_usage(job["client_id"], job["recipe"])
    return {"job_id": job_id, "status": "complete", "result": payload["result"]}


@app.delete("/api/jobs/{job_id}")
def cancel_analysis_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Analysis job not found or expired")
        process = job["process"]
        job["cancelled"] = True
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1)
        Path(job["output"]).unlink(missing_ok=True)
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/usage")
def usage_summary():
    return _usage_summary()


@app.get("/api/citation")
def citation():
    return {"version": "1.2.1", "preferred": "KofiOsae (2026). SEM Metrology Workbench (Version 1.2.1) [Computer software]. https://github.com/KofiOsae/sem-metrology-workbench",
            "cff": "https://github.com/KofiOsae/sem-metrology-workbench/blob/main/CITATION.cff",
            "license": "BSD-3-Clause", "copyright": "Copyright © 2026 KofiOsae"}
