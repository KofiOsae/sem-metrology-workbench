from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from sem_metrology.calibration import detect_footer
from sem_metrology.io import to_gray
from sem_metrology.lines import analyze_line_space
from sem_metrology.measurements import measure_cross_section, measure_layer_stack
from sem_metrology.overlay import circle_overlay, cross_section_overlay, line_space_overlay, rectangular_overlay, shape_overlay
from sem_metrology.reporting import measurement_statistics
from sem_metrology.shapes import analyze_circular_outlines, analyze_rectangular_features, analyze_shapes
from sem_metrology.waveguide import analyze_waveguide, waveguide_overlay

APP_VERSION = "1.2.1-web"
ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_UPLOAD = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000

app = FastAPI(title="SEM Metrology Workbench", version=APP_VERSION)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


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
    return JSONResponse({"width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                         "preview": _overlay_data(rgb)})


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...), recipe: str = Form(...), nm_per_px: float = Form(...),
    min_size_nm: float = Form(10), max_size_nm: float = Form(10000), sensitivity: float = Form(.82),
    polarity: str = Form("bright"), orientation: str = Form("vertical"), reject_border: bool = Form(True),
    reference_roi: str = Form(""), analysis_roi: str = Form("")
):
    raw=await image.read(MAX_UPLOAD+1)
    if len(raw)>MAX_UPLOAD:raise HTTPException(413,"Image exceeds the 20 MB browser limit")
    try: rgb=np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"),dtype=np.uint8)
    except Exception as exc:raise HTTPException(400,f"Cannot read image: {exc}")
    if rgb.shape[0]*rgb.shape[1]>MAX_PIXELS:raise HTTPException(413,"Image exceeds the 25 megapixel browser limit")
    if not np.isfinite(nm_per_px) or nm_per_px<=0:raise HTTPException(400,"Enter a positive calibrated nm/pixel value")
    footer=detect_footer(rgb); analysis=rgb[:footer].copy() if footer<rgb.shape[0] else rgb.copy(); gray=to_gray(analysis); scale=nm_per_px/1000.0
    ref=_roi(reference_roi,gray.shape[1],gray.shape[0]); roi=_roi(analysis_roi,gray.shape[1],gray.shape[0])
    rows=[]; measurements=[]; overlay=analysis.copy(); warnings=[]
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
        rows,_,contours=analyze_shapes(crop,scale,polarity,min_area,max_area,reject_border,nominal,sensitivity,template)
        local=np.asarray(shape_overlay(analysis[y:y+h,x:x+w],contours,rows)); overlay[y:y+h,x:x+w]=local
        if any(bool(row.get("Contact split")) for row in rows):warnings.append("Touching particles are visible-segment estimates; hidden boundaries are not reconstructed.")
    elif recipe=="grating":
        rows,_,bands=analyze_line_space(gray,scale,polarity,min_um,max_um,orientation); overlay=np.asarray(line_space_overlay(analysis,bands,orientation))
    elif recipe in ("waveguide","cross_section","layer_stack"):
        if roi is None:raise HTTPException(400,"Draw an analysis ROI around the intended structure")
        x,y,w,h=roi; crop=gray[y:y+h,x:x+w]; crop_rgb=analysis[y:y+h,x:x+w]
        if recipe=="waveguide":
            rows,diagnostics=analyze_waveguide(crop,scale); overlay[y:y+h,x:x+w]=waveguide_overlay(crop_rgb,diagnostics)
        elif recipe=="cross_section":
            measurements,diagnostics=measure_cross_section(crop,scale); overlay[y:y+h,x:x+w]=np.asarray(cross_section_overlay(crop_rgb,diagnostics))
        else:
            measurements,diagnostics=measure_layer_stack(crop,scale); overlay[y:y+h,x:x+w]=np.asarray(cross_section_overlay(crop_rgb,diagnostics)); warnings.append("Layer interfaces are automatic candidates and require operator review in the desktop edition.")
    else:raise HTTPException(400,"Unknown recipe")
    stats=measurement_statistics(measurements) if measurements else _numeric_stats(rows)
    if not rows and not measurements:warnings.append("No credible structures were found. Review calibration, polarity, size limits, ROI and recipe.")
    return JSONResponse(_safe({"version":APP_VERSION,"recipe":recipe,"feature_count":len(rows) if rows else (1 if measurements else 0),"rows":rows,"statistics":stats,"warnings":warnings,"overlay":_overlay_data(overlay),"privacy":"Image processed in memory and not intentionally retained by the application."}))
