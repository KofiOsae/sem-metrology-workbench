from __future__ import annotations

import cv2
import numpy as np

from .roughness import low_pass_three_point, spatial_descriptors


def estimate_line_direction(gray: np.ndarray) -> tuple[float,float]:
    """Estimate manufactured line direction in image degrees (0 horizontal, 90 vertical)."""
    specimen=gray[:max(20,int(gray.shape[0]*.84))]
    tensor_work=cv2.GaussianBlur(specimen,(3,3),0); gx=cv2.Sobel(tensor_work,cv2.CV_32F,1,0); gy=cv2.Sobel(tensor_work,cv2.CV_32F,0,1)
    jxx,jyy,jxy=float(np.mean(gx*gx)),float(np.mean(gy*gy)),float(np.mean(gx*gy))
    tensor_normal=float((.5*np.degrees(np.arctan2(2*jxy,jxx-jyy)))%180); tensor_direction=(tensor_normal+90)%180
    tensor_conf=float(np.hypot(jxx-jyy,2*jxy)/max(jxx+jyy,1e-12))
    edges=cv2.Canny(cv2.GaussianBlur(specimen,(5,5),0),40,120)
    segments=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(25,min(gray.shape)//12),minLineLength=max(20,min(gray.shape)//8),maxLineGap=12)
    if segments is None: return (tensor_direction,tensor_conf) if tensor_conf>=.06 else (np.nan,0.0)
    angles=[]; weights=[]
    for x1,y1,x2,y2 in np.asarray(segments).reshape((-1,4)):
        length=float(np.hypot(x2-x1,y2-y1))
        if length<=0: continue
        angle=float(np.degrees(np.arctan2(y2-y1,x2-x1))%180); angles.append(angle); weights.append(length)
    if not angles: return (tensor_direction,tensor_conf) if tensor_conf>=.06 else (np.nan,0.0)
    # doubled-angle circular mean handles the equivalence of θ and θ+180°.
    radians=np.radians(np.asarray(angles)*2); vector=np.sum(np.asarray(weights)*np.exp(1j*radians))
    direction=float((np.degrees(np.angle(vector))/2)%180); confidence=float(abs(vector)/max(np.sum(weights),1e-12))
    disagreement=abs(((direction-tensor_direction+90)%180)-90)
    if tensor_conf>=.10 and disagreement>15: return tensor_direction,tensor_conf
    return direction,confidence


def analyze_line_space(
    gray: np.ndarray,
    um_per_px: float,
    polarity: str = "bright",
    min_cd_um: float = 0.02,
    max_cd_um: float = 10.0,
    orientation: str = "vertical",
) -> tuple[list[dict], np.ndarray, list[tuple[int, int]]]:
    """Measure effectively infinite vertical or horizontal grating lines."""
    transposed = orientation == "horizontal"
    if transposed:
        gray = gray.T
    work = cv2.GaussianBlur(gray, (3, 3), 0)
    profile = work.mean(axis=0).astype(np.uint8)
    threshold, _ = cv2.threshold(profile, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = work > threshold
    mask = bright if polarity == "bright" else ~bright
    edge_strength = np.abs(cv2.Sobel(work,cv2.CV_32F,1,0,ksize=3))
    occupancy = mask.mean(axis=0)
    nominal = occupancy > 0.50
    changes = np.diff(np.r_[False, nominal, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    min_px = max(2, min_cd_um / um_per_px)
    max_px = max(min_px, max_cd_um / um_per_px)
    bands = [(int(a), int(b - 1)) for a, b in zip(starts, ends) if min_px <= b - a <= max_px]
    results = []
    valid_bands = []
    for band_index, (left0, right0) in enumerate(bands, start=1):
        margin = max(3, int((right0 - left0 + 1) * 0.5))
        lefts, rights = [], []
        for gradient_row in edge_strength:
            llo,lhi=max(0,left0-margin),min(len(gradient_row),left0+margin+1)
            rlo,rhi=max(0,right0-margin),min(len(gradient_row),right0+margin+1)
            if lhi-llo>=3 and rhi-rlo>=3:
                lefts.append(float(llo+np.argmax(gradient_row[llo:lhi]))); rights.append(float(rlo+np.argmax(gradient_row[rlo:rhi])))
        if len(lefts) < gray.shape[0] * 0.60:
            continue
        lefts, rights = np.asarray(lefts), np.asarray(rights)
        widths = rights - lefts + 1
        y = np.arange(len(lefts), dtype=float)
        left_residual = lefts - np.polyval(np.polyfit(y, lefts, 1), y)
        right_residual = rights - np.polyval(np.polyfit(y, rights, 1), y)
        width_residual = widths-widths.mean()
        left_filtered=low_pass_three_point(left_residual); right_filtered=low_pass_three_point(right_residual); width_filtered=low_pass_three_point(width_residual)
        left_corr,left_wave=spatial_descriptors(left_residual,um_per_px); right_corr,right_wave=spatial_descriptors(right_residual,um_per_px); width_corr,width_wave=spatial_descriptors(width_residual,um_per_px)
        mean_cd = float(widths.mean() * um_per_px)
        results.append({
            "Band": band_index,
            "N rows": len(widths),
            "Mean CD (µm)": mean_cd,
            "CD Std (µm)": float(widths.std(ddof=1) * um_per_px),
            "CD Min (µm)": float(widths.min() * um_per_px),
            "CD Max (µm)": float(widths.max() * um_per_px),
            "LWR 3σ (µm)": float(3 * widths.std(ddof=1) * um_per_px),
            "LWR RMS (µm)": float(width_residual.std(ddof=1)*um_per_px),
            "Left LER 3σ (µm)": float(3 * left_residual.std(ddof=1) * um_per_px),
            "Right LER 3σ (µm)": float(3 * right_residual.std(ddof=1) * um_per_px),
            "Left LER RMS (µm)": float(left_residual.std(ddof=1)*um_per_px),
            "Right LER RMS (µm)": float(right_residual.std(ddof=1)*um_per_px),
            "Filtered LWR 3σ (µm)": float(3*width_filtered.std(ddof=1)*um_per_px),
            "Filtered left LER 3σ (µm)": float(3*left_filtered.std(ddof=1)*um_per_px),
            "Filtered right LER 3σ (µm)": float(3*right_filtered.std(ddof=1)*um_per_px),
            "LWR correlation length (µm)": width_corr,
            "Left LER correlation length (µm)": left_corr,
            "Right LER correlation length (µm)": right_corr,
            "LWR dominant wavelength (µm)": width_wave,
            "Left LER dominant wavelength (µm)": left_wave,
            "Right LER dominant wavelength (µm)": right_wave,
            "Sampling length (µm)": float(len(widths)*um_per_px),
            "Sampling interval (µm)": float(um_per_px),
            "Roughness method": "gradient peak; linear detrend; raw plus disclosed 3-point [1,2,1]/4 low-pass; SEM-noise inclusive",
            "Center X (px)": float(((lefts + rights) / 2).mean()),
            "Orientation": orientation,
        })
        valid_bands.append({"left":int(round(lefts.mean())),"right":int(round(rights.mean())),"left_trace":lefts.tolist(),"right_trace":rights.tolist(),"samples":len(lefts)})
    centers = np.asarray([row["Center X (px)"] for row in results])
    if len(centers) > 1:
        pitches = np.diff(centers) * um_per_px
        for index, pitch in enumerate(pitches):
            results[index]["Pitch to next (µm)"] = float(pitch)
            results[index]["Duty cycle (%)"] = float(results[index]["Mean CD (µm)"]/pitch*100) if pitch else np.nan
        results[-1]["Pitch to next (µm)"] = np.nan
        results[-1]["Duty cycle (%)"] = np.nan
    else:
        for row in results:
            row["Pitch to next (µm)"] = np.nan
            row["Duty cycle (%)"] = np.nan
    output_mask = mask.astype(np.uint8) * 255
    return results, output_mask.T if transposed else output_mask, valid_bands
