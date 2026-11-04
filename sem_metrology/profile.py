from __future__ import annotations

import cv2
import numpy as np


def _crossing(x: np.ndarray, y: np.ndarray, start: int, stop: int, level: float, reverse: bool) -> float:
    indices=range(stop-1,start-1,-1) if reverse else range(start,stop)
    previous=None
    for index in indices:
        if previous is not None:
            a,b=previous,index
            ya,yb=float(y[a]-level),float(y[b]-level)
            if ya==0: return float(x[a])
            if ya*yb<=0 and yb!=ya:
                fraction=-ya/(yb-ya)
                return float(x[a]+fraction*(x[b]-x[a]))
        previous=index
    return float(x[start] if reverse else x[stop-1])


def measure_lateral_profile(signal: np.ndarray, sampling_per_px: float, expected_min_px: float=8.0,
                            polarity: str="auto") -> tuple[list[dict],dict]:
    """Measure calibrated lateral chords from a one-dimensional SEM signal.

    Edges are local half-prominence crossings (FWHM-like), not topographic
    heights. A chord equals particle diameter only for a central transect of an
    approximately circular particle.
    """
    raw=np.asarray(signal,dtype=float).ravel()
    if len(raw)<12: return [],{"reason":"profile too short"}
    coordinate=np.arange(len(raw),dtype=float)
    # Remove only slow illumination drift; retain particle-scale contrast.
    trend=np.polyval(np.polyfit(coordinate,raw,1),coordinate)
    detrended=raw-trend+float(np.mean(trend))
    smooth_px=max(1.0,min(8.0,float(expected_min_px)/8.0))
    kernel=max(5,int(round(6*smooth_px+1))); kernel+=1-kernel%2
    smooth=cv2.GaussianBlur(detrended.astype(np.float32).reshape(1,-1),(kernel,1),smooth_px).ravel().astype(float)
    selected=polarity.lower().split()[0]
    if selected not in ("bright","dark"):
        median=float(np.median(smooth)); upper=float(np.percentile(smooth,95))-median; lower=median-float(np.percentile(smooth,5))
        selected="bright" if upper>=lower else "dark"
    working=smooth if selected=="bright" else -smooth
    noise=float(1.4826*np.median(np.abs(np.diff(working)-np.median(np.diff(working))))/np.sqrt(2))
    amplitude=float(np.percentile(working,95)-np.percentile(working,20))
    prominence_gate=max(2.5*noise,.10*amplitude,1.0)
    # A dilated local-maximum mask handles the broad, nearly flat tops that are
    # common when a line crosses a particle. Collapse every maximum plateau to
    # one representative instead of depending on a strict adjacent-pixel peak.
    nms_window=max(3,int(round(.35*expected_min_px))); nms_window+=1-nms_window%2
    dilated=cv2.dilate(working.astype(np.float32).reshape(1,-1),np.ones((1,nms_window),np.uint8)).ravel()
    maximum_mask=working>=dilated.astype(float)-max(1e-6,noise*.02)
    labels_count,labels=cv2.connectedComponents(maximum_mask.astype(np.uint8).reshape(1,-1))
    candidates=[]
    for label in range(1,labels_count):
        members=np.where(labels.ravel()==label)[0]
        if len(members): candidates.append(int(members[np.argmax(working[members])]))
    candidates=sorted(candidates,key=lambda index:working[index],reverse=True)
    min_distance=max(3,int(round(.35*expected_min_px))); peaks=[]
    for index in candidates:
        # Search beyond the minimum credible chord: the actual particle can be
        # substantially wider than that operator limit.
        radius=max(min_distance,int(round(2.5*expected_min_px)))
        left=max(0,index-radius); right=min(len(working)-1,index+radius)
        local_base=max(float(np.min(working[left:index+1])),float(np.min(working[index:right+1])))
        prominence=float(working[index]-local_base)
        if prominence<prominence_gate: continue
        if any(abs(index-old)<min_distance for old in peaks): continue
        peaks.append(int(index))
    peaks.sort(); rows=[]
    for order,index in enumerate(peaks):
        left_limit=0 if order==0 else int(round((peaks[order-1]+index)/2))
        right_limit=len(working)-1 if order==len(peaks)-1 else int(round((index+peaks[order+1])/2))
        left_base=float(np.min(working[left_limit:index+1])); right_base=float(np.min(working[index:right_limit+1]))
        baseline=max(left_base,right_base); prominence=float(working[index]-baseline)
        level=baseline+.5*prominence
        left_px=_crossing(coordinate,working,left_limit,index+1,level,True)
        right_px=_crossing(coordinate,working,index,right_limit+1,level,False)
        chord_px=right_px-left_px
        if chord_px<max(2,.18*expected_min_px): continue
        confidence=float(np.clip((prominence/max(prominence_gate,1e-9)-1)/2+.55,.35,.98))
        rows.append({"Chord":len(rows)+1,"Left edge (px)":left_px,"Right edge (px)":right_px,
                     "Center position":.5*(left_px+right_px)*sampling_per_px,
                     "Lateral chord":chord_px*sampling_per_px,"Peak prominence (intensity)":prominence,
                     "Edge level (intensity)":level if selected=="bright" else -level,
                     "Detection confidence (%)":100*confidence,"Polarity":selected})
    for index,row in enumerate(rows):
        if index+1<len(rows):
            row["Center spacing"]=rows[index+1]["Center position"]-row["Center position"]
            row["Edge-to-edge gap"]=(rows[index+1]["Left edge (px)"]-row["Right edge (px)"])*sampling_per_px
        else: row["Center spacing"]=np.nan; row["Edge-to-edge gap"]=np.nan
    coverage=float(sum(row["Lateral chord"] for row in rows)/(max(1,len(raw)-1)*sampling_per_px)*100)
    return rows,{"polarity":selected,"noise":noise,"prominence_gate":prominence_gate,"coverage_percent":coverage,
                 "smooth_signal":smooth,"peak_indices":[int(round(row["Center position"]/sampling_per_px)) for row in rows]}
