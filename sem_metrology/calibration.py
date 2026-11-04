from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from pathlib import Path
import re

import cv2
import numpy as np


def read_sem_companion_metadata(image_path: str | Path) -> dict:
    """Read common Hitachi-style sidecar metadata without trusting raster text."""
    path=Path(image_path); sidecar=path.with_suffix(".txt")
    if not sidecar.exists(): return {}
    try:
        raw=sidecar.read_bytes()
        text=raw.decode("utf-16-le",errors="ignore") if raw[:200].count(b"\x00")>20 else raw.decode("utf-8",errors="ignore")
    except OSError: return {}
    result={"sidecar":str(sidecar)}
    for key in ("PixelSize","DataSize","MicronMarker","RasterRotation","Magnification","InstructName","SignalName"):
        match=re.search(rf"(?m)^{re.escape(key)}=(.*)$",text)
        if match: result[key]=match.group(1).strip()
    try: result["pixel_size_nm"]=float(result["PixelSize"])
    except (KeyError,ValueError): pass
    return result


@dataclass(frozen=True)
class Calibration:
    microns_per_pixel: float
    known_length_um: float
    pixel_length: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    source: str = "confirmed_scale_bar"
    confirmed: bool = True

    @classmethod
    def from_line(
        cls,
        known_length_um: float,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        source: str = "confirmed_scale_bar",
    ) -> "Calibration":
        pixels = hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
        if known_length_um <= 0 or pixels <= 0:
            raise ValueError("Known length and pixel length must both be positive")
        return cls(known_length_um / pixels, known_length_um, pixels, start_xy, end_xy, source)

    def as_dict(self) -> dict:
        return asdict(self)


def detect_footer(rgb: np.ndarray) -> int:
    """Return the likely first footer row, or the image height if none is found."""
    height = rgb.shape[0]
    gray = rgb.mean(axis=2)
    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    lo = int(height * 0.65)
    # SEM exports may switch to either a bright ZEISS-style or dark Hitachi-style
    # information strip. Use the magnitude of the mean transition while still
    # favoring the accompanying loss of specimen texture.
    # A footer boundary is normally a broad row-wise transition. Penalize rows
    # with high lateral variation so a short bright scale bar cannot masquerade
    # as the full-width boundary.
    score = (np.abs(np.diff(row_mean)) + .20*np.maximum(0,-np.diff(row_std))
             - .60*np.maximum(row_std[:-1],row_std[1:]))
    # Exclude the very bottom ruler/text rows from boundary selection. If an
    # unusually shallow footer has no transition in this band, fall back to the
    # full lower image.
    search_end=max(lo+1,int(height*.93))
    candidates = np.arange(lo, search_end)
    if not len(candidates):
        return height
    lower_score=score[lo:search_end]; peak=float(np.max(lower_score))
    if peak < 20:
        candidates=np.arange(lo,height-1); lower_score=score[lo:height-1]; peak=float(np.max(lower_score))
        if peak<20:return height
    idx=int(candidates[int(np.argmax(lower_score))])+1
    return idx


def detect_scale_bar(rgb: np.ndarray, footer_start: int | None = None):
    """Detect the green horizontal ZEISS scale-bar segment.

    Returns endpoints and a confidence score. The text value deliberately remains
    operator-confirmed because it is raster text, not calibrated TIFF metadata.
    """
    h, w, _ = rgb.shape
    footer_start = detect_footer(rgb) if footer_start is None else footer_start
    # Rulers are often printed just above the text footer boundary.
    y0 = int(h*.68) if int(footer_start)>=h else max(0,int(footer_start)-int(h*.12))
    # Some SEM exports have no sharply separated footer. Their white scale bar
    # still sits in the lower information band.
    if y0 >= h: y0=int(h*.68)
    roi = rgb[y0:, :, :]
    if roi.size == 0:
        return None
    r, g, b = [roi[..., i].astype(int) for i in range(3)]
    green = (g > 100) & (g > r * 1.35 + 15) & (g > b * 1.20 + 10)
    # Find the row containing the longest green horizontal run.
    best = None
    for yy in range(green.shape[0]):
        xs = np.flatnonzero(green[yy])
        if xs.size < 5:
            continue
        split = np.where(np.diff(xs) > 2)[0] + 1
        for run in np.split(xs, split):
            if run.size < 5:
                continue
            length = int(run[-1] - run[0])
            if best is None or length > best[0]:
                best = (length, int(run[0]), int(run[-1]), yy + y0)
    if best is None:
        # White/gray tick-ended scale bars are common in Hitachi exports. Search
        # only the lower band and require vertical tick support, which rejects
        # most footer text baselines and instrument metadata.
        gray=roi.mean(axis=2)
        bright=gray>=max(165,float(np.percentile(gray,96)))
        white_best=None
        for yy in range(2,bright.shape[0]-3):
            xs=np.flatnonzero(bright[yy])
            if xs.size<8: continue
            split=np.where(np.diff(xs)>4)[0]+1
            for run in np.split(xs,split):
                if run.size<8: continue
                x1,x2=int(run[0]),int(run[-1]); length=x2-x1
                if length<w*.04 or length>w*.65: continue
                columns=np.arange(x1,x2+1)
                vertical=np.sum(bright[max(0,yy-5):min(bright.shape[0],yy+6),columns],axis=0)
                tick_columns=int(np.sum(vertical>=4))
                if tick_columns<2: continue
                # Determine the physical datum at the centres of the end ticks,
                # excluding antialiasing/round line caps beyond those centres.
                upper=np.sum(bright[max(0,yy-9):max(0,yy-3),columns],axis=0)
                lower=np.sum(bright[min(bright.shape[0],yy+4):min(bright.shape[0],yy+10),columns],axis=0)
                tick_x=columns[(upper+lower)>=3]; tick_runs=[]
                if len(tick_x):
                    tick_runs=[part for part in np.split(tick_x,np.where(np.diff(tick_x)>1)[0]+1) if len(part)]
                datum_x1,datum_x2=x1,x2
                if len(tick_runs)>=2:
                    left_center=float(np.mean(tick_runs[0])); right_center=float(np.mean(tick_runs[-1]))
                    if right_center-left_center>=w*.04: datum_x1,datum_x2=round(left_center),round(right_center)
                datum_length=datum_x2-datum_x1
                score=datum_length*(1+min(tick_columns,12)/12)
                if white_best is None or score>white_best[0]:
                    white_best=(score,datum_length,datum_x1,datum_x2,yy+y0,tick_columns)
        # Some Hitachi rulers consist only of a regular train of short vertical
        # ticks, with no horizontal baseline.
        count,_,stats,centroids=cv2.connectedComponentsWithStats(bright.astype(np.uint8),8)
        ticks=[]
        for label in range(1,count):
            x,y,width,height,area=[int(v) for v in stats[label]]
            if 1<=width<=6 and 3<=height<=16 and area>=3:
                ticks.append((float(centroids[label,0]),float(centroids[label,1]),height))
        ruler_best=None
        for _,anchor_y,_ in ticks:
            group=sorted([tick for tick in ticks if abs(tick[1]-anchor_y)<=3])
            xs=np.array([tick[0] for tick in group])
            if len(xs)<4: continue
            gaps=np.diff(xs); plausible=gaps[(gaps>=4)&(gaps<=w*.12)]
            if len(plausible)<3: continue
            spacing=float(np.median(plausible))
            chain=[float(xs[0])]
            for value in xs[1:]:
                if .65*spacing<=value-chain[-1]<=1.35*spacing: chain.append(float(value))
                elif len(chain)<4: chain=[float(value)]
            if len(chain)<4: continue
            length=chain[-1]-chain[0]
            regularity=float(np.std(np.diff(chain))/(spacing+1e-9))
            score=length*(1- min(regularity,1))
            if length>=w*.05 and (ruler_best is None or score>ruler_best[0]):
                ruler_best=(score,chain,anchor_y,regularity)
        if ruler_best is None:
            if white_best is None: return None
            _,length,x1,x2,y,ticks=white_best
            confidence=min(.9,.45+length/max(1,w)*1.5+min(ticks,8)*.025)
            return {"start_xy":(x1,y),"end_xy":(x2,y),"pixel_length":length,
                    "confidence":round(confidence,3),"footer_start":footer_start,
                    "style":"white_tick_bar"}
        _,chain,anchor_y,regularity=ruler_best
        if white_best is not None and white_best[1] > (chain[-1]-chain[0])*1.35:
            _,length,x1,x2,y,ticks=white_best
            confidence=min(.9,.45+length/max(1,w)*1.5+min(ticks,8)*.025)
            return {"start_xy":(x1,y),"end_xy":(x2,y),"pixel_length":length,
                    "confidence":round(confidence,3),"footer_start":footer_start,
                    "style":"white_tick_bar"}
        confidence=float(np.clip(.55+.04*len(chain)-.25*regularity,.45,.9))
        return {"start_xy":(round(chain[0]),round(anchor_y+y0)),
                "end_xy":(round(chain[-1]),round(anchor_y+y0)),
                "pixel_length":round(chain[-1]-chain[0]),
                "confidence":round(confidence,3),"footer_start":footer_start,
                "style":"white_tick_ruler","tick_count":len(chain)}
    length, x1, x2, y = best
    confidence = min(1.0, length / max(20.0, w * 0.08))
    return {"start_xy": (x1, y), "end_xy": (x2, y), "pixel_length": length,
            "confidence": round(confidence, 3), "footer_start": footer_start,
            "style":"green_bar"}
