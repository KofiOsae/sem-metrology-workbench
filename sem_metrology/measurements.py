from __future__ import annotations

import numpy as np
import cv2

from .models import Measurement


def measure_lines(lines: list[dict], um_per_px: float) -> list[Measurement]:
    """Convert operator-reviewed endpoint pairs into traceable distances."""
    results = []
    for index, line in enumerate(lines, start=1):
        try:
            x1, y1 = float(line["x1"]), float(line["y1"])
            x2, y2 = float(line["x2"]), float(line["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        pixels = float(np.hypot(x2 - x1, y2 - y1))
        if pixels <= 0:
            continue
        label = str(line.get("label") or f"Distance {index}")
        results.append(Measurement(
            label, pixels * um_per_px, "µm", "operator-confirmed endpoints", 1.0,
            {"pixels": pixels, "start_xy": [x1, y1], "end_xy": [x2, y2]},
        ))
    return results


def _otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    best, threshold = -1.0, 127
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        between = weight_bg * weight_fg * (sum_bg / weight_bg - (sum_total - sum_bg) / weight_fg) ** 2
        if between > best:
            best, threshold = between, t
    return threshold


def _autocorrelation_pitch(profile: np.ndarray, min_period_px: int = 5):
    signal = profile.astype(float) - profile.mean()
    corr = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
    if corr[0] <= 0 or len(corr) <= min_period_px + 2:
        return None, 0.0
    corr /= corr[0]
    # First meaningful local maximum after the exclusion zone.
    peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] >= corr[2:]))[0] + 1
    peaks = peaks[peaks >= min_period_px]
    if not len(peaks):
        return None, 0.0
    strong = peaks[corr[peaks] > 0.12]
    lag = int(strong[0] if len(strong) else peaks[np.argmax(corr[peaks])])
    return lag, float(np.clip(corr[lag], 0, 1))


def measure_top_view(gray: np.ndarray, um_per_px: float) -> tuple[list[Measurement], dict]:
    """Measure repeated pitch across multiple strips, not from one global FFT value."""
    threshold = _otsu(gray)
    # Strict comparison matters when Otsu selects the background mode (often 0).
    binary = gray > threshold
    results: list[Measurement] = []
    diagnostics = {"threshold": threshold, "binary": binary}
    strip_count = 12
    for axis in ("x", "y"):
        pitch_px, confidences = [], []
        strip_source = gray if axis == "x" else gray.T
        for strip in np.array_split(strip_source, strip_count, axis=0):
            if not len(strip):
                continue
            lag, confidence = _autocorrelation_pitch(strip.mean(axis=0))
            if lag and confidence >= 0.20:
                pitch_px.append(float(lag))
                confidences.append(confidence)
        if len(pitch_px) >= 3:
            values = np.asarray(pitch_px) * um_per_px
            mean = float(values.mean())
            results.append(Measurement(
                f"Mean pitch {axis.upper()}", mean, "µm",
                "multi-strip normalized autocorrelation", float(np.mean(confidences)),
                {"n": len(values), "std_um": float(values.std(ddof=1)),
                 "median_um": float(np.median(values)), "min_um": float(values.min()),
                 "max_um": float(values.max()),
                 "cv_percent": float(values.std(ddof=1) / mean * 100) if mean else 0.0},
            ))
    duty = float(binary.mean())
    results.append(Measurement("Bright area fraction", duty * 100, "%", "Otsu segmentation", 0.6))
    return results, diagnostics


def measure_cross_section(gray: np.ndarray, um_per_px: float) -> tuple[list[Measurement], dict]:
    """Measure a bright cross-section feature within an operator-selected ROI."""
    threshold = _otsu(gray)
    mask = gray > threshold
    # Retain rows/columns occupied by a substantial bright feature, suppressing speckle.
    row_min = max(2, int(gray.shape[1] * 0.03))
    ys = np.flatnonzero(mask.sum(axis=1) >= row_min)
    if not len(ys):
        return [], {"threshold": threshold, "mask": mask, "reason": "no feature"}
    y_top, raw_bottom = int(ys[0]), int(ys[-1])
    # A cross-cut pillar commonly merges into a horizontal substrate. Locate a
    # laterally persistent horizontal transition instead of using only feature
    # occupancy: a tapered sidewall can cross an occupancy threshold well above
    # the true substrate and would systematically under-report height.
    occupancy_by_row=mask.mean(axis=1); base_rows=np.flatnonzero(occupancy_by_row>=0.65)
    substrate_top=None
    minimum_base_row=y_top+max(5,int(0.45*(raw_bottom-y_top)))
    vertical_change=np.abs(np.diff(gray.astype(np.float32),axis=0))
    robust_noise=float(np.median(vertical_change)+1.4826*np.median(np.abs(vertical_change-np.median(vertical_change))))
    row_strength=np.mean(vertical_change,axis=1)
    row_support=np.mean(vertical_change>max(8,2.5*robust_noise),axis=1)
    search=np.arange(max(0,minimum_base_row-1),min(raw_bottom,len(row_strength)))
    if len(search):
        candidate=int(search[np.argmax(row_strength[search])])+1
        if row_strength[candidate-1]>=max(6,2.5*np.median(row_strength[search])) and row_support[candidate-1]>=.50:
            substrate_top=candidate
    # Conservative fallback for low-contrast substrates with no coherent
    # gradient; require a sustained occupancy jump, not a single wide row.
    if substrate_top is None:
        for candidate in base_rows:
            before=occupancy_by_row[max(y_top,candidate-3):candidate]
            after=occupancy_by_row[candidate:min(len(occupancy_by_row),candidate+3)]
            if candidate>=minimum_base_row and len(before) and len(after) and np.mean(after)-np.mean(before)>=.18 and np.mean(after)>=.65:
                substrate_top=int(candidate); break
    y_bottom=(substrate_top-1) if substrate_top is not None else raw_bottom
    if y_bottom<=y_top: y_bottom=raw_bottom; substrate_top=None
    height_px = y_bottom - y_top + 1
    widths = {}
    edge_points = []
    for fraction in (0.1, 0.5, 0.9):
        y = int(round(y_top + fraction * max(0, height_px - 1)))
        xs = np.flatnonzero(mask[y])
        if len(xs):
            widths[fraction] = int(xs[-1] - xs[0] + 1)
            edge_points.append((int(xs[0]), int(xs[-1]), y))
    occupancy = mask[y_top:y_bottom + 1].mean()
    confidence = float(np.clip(occupancy * 2, 0.25, 0.95))
    results = [Measurement("Feature height", height_px * um_per_px, "µm",
                           "threshold edge span", confidence, {"pixels": height_px})]
    labels = {0.1: "Top width", 0.5: "Middle width", 0.9: "Bottom width"}
    for fraction, width_px in widths.items():
        results.append(Measurement(labels[fraction], width_px * um_per_px, "µm",
                                   "row edge span", confidence, {"pixels": width_px}))
    if 0.1 in widths and 0.9 in widths and height_px > 1:
        run = abs(widths[0.9] - widths[0.1]) / 2
        angle = np.degrees(np.arctan2(height_px * 0.8, max(run, 1e-9)))
        results.append(Measurement("Nominal sidewall angle", float(angle), "°",
                                   "top-bottom width model", confidence * 0.8))
    trace_y=[]; left_trace=[]; right_trace=[]
    for y in range(y_top,y_bottom+1):
        xs=np.flatnonzero(mask[y])
        if len(xs)>=2: trace_y.append(y); left_trace.append(float(xs[0])); right_trace.append(float(xs[-1]))
    if len(trace_y)>=8:
        trace_y=np.asarray(trace_y,float); left_trace=np.asarray(left_trace); right_trace=np.asarray(right_trace)
        left_fit=np.polyval(np.polyfit(trace_y,left_trace,1),trace_y); right_fit=np.polyval(np.polyfit(trace_y,right_trace,1),trace_y)
        left_slope=float(np.polyfit(trace_y,left_trace,1)[0]); right_slope=float(np.polyfit(trace_y,right_trace,1)[0])
        left_angle=float(np.degrees(np.arctan2(1,abs(left_slope)))); right_angle=float(np.degrees(np.arctan2(1,abs(right_slope))))
        left_residual=left_trace-left_fit; right_residual=right_trace-right_fit
        results += [Measurement("Left sidewall angle",left_angle,"°","linear fit to continuous sidewall edge",confidence*.8),Measurement("Right sidewall angle",right_angle,"°","linear fit to continuous sidewall edge",confidence*.8),Measurement("Left sidewall roughness 3σ",float(3*left_residual.std(ddof=1)*um_per_px),"µm","edge residual from fitted sidewall; noise inclusive",confidence*.7),Measurement("Right sidewall roughness 3σ",float(3*right_residual.std(ddof=1)*um_per_px),"µm","edge residual from fitted sidewall; noise inclusive",confidence*.7)]
        sidewall_traces=(np.column_stack((left_trace,trace_y)),np.column_stack((right_trace,trace_y)))
    else: sidewall_traces=()
    profile=cv2.GaussianBlur(gray,(5,5),0).mean(axis=1); gradient=np.abs(np.gradient(profile)); gate=float(np.percentile(gradient,90)); peaks=[]
    for index in np.argsort(gradient)[::-1]:
        if gradient[index]<gate or any(abs(int(index)-old)<4 for old in peaks): continue
        peaks.append(int(index))
        if len(peaks)>=8: break
    interface_rows=sorted(peaks)
    for index,(upper,lower) in enumerate(zip(interface_rows,interface_rows[1:]),start=1):
        results.append(Measurement(f"Candidate layer thickness {index}",(lower-upper)*um_per_px,"µm","vertical mean-signal gradient interfaces",min(confidence*.65,.7),{"upper_row":upper,"lower_row":lower}))
    center_x=int(round(np.median([(left+right)/2 for left,right,_ in edge_points]))) if edge_points else gray.shape[1]//2
    return results, {"threshold": threshold, "mask": mask, "edge_points": edge_points,
                     "bounds": (y_top, y_bottom),"sidewall_traces":sidewall_traces,"interface_rows":interface_rows,
                     "height_line":(center_x,y_top,center_x,(substrate_top if substrate_top is not None else y_bottom)),"height_datum":"feature top to local substrate interface" if substrate_top is not None else "feature top to lowest detected feature row"}


def measure_layer_stack(gray: np.ndarray, um_per_px: float,
                        interface_rows: list[int] | None = None) -> tuple[list[Measurement],dict]:
    """Measure a film stack from reviewed vertical signal transitions.

    ``interface_rows`` is the operator-reviewed interface set.  When omitted the
    candidates are proposed from the laterally averaged vertical gradient.
    """
    blurred=cv2.GaussianBlur(gray,(7,7),0).astype(float)
    smooth=blurred.mean(axis=1)
    vertical_edges=np.abs(np.gradient(blurred,axis=0))
    gradient=np.abs(np.gradient(smooth))
    edge_gate=float(np.percentile(vertical_edges,88))
    # Real film boundaries persist laterally. Internal texture may have a strong
    # local gradient but usually does not align across much of the ROI width.
    support=np.array([np.mean(np.max(vertical_edges[max(0,row-2):min(len(smooth),row+3)],axis=0)>=edge_gate)
                      for row in range(len(smooth))])
    strength=gradient/(np.percentile(gradient,95)+1e-9)
    candidate_score=np.clip(strength,0,2)*np.sqrt(np.clip(support,0,1))
    gate=max(float(np.percentile(candidate_score,86)),.28)
    if interface_rows is None:
        peaks=[]
        for index in np.argsort(candidate_score)[::-1]:
            if candidate_score[index]<gate or support[index]<.28 or index<2 or index>len(gradient)-3: continue
            # Transitions closer than eight source pixels are normally the two
            # sides of one blurred/haloed SEM edge, not distinct material layers.
            if any(abs(int(index)-old)<8 for old in peaks): continue
            peaks.append(int(index))
            if len(peaks)>=7: break
        interfaces=sorted(peaks)
    else:
        interfaces=sorted({int(np.clip(row,2,len(gradient)-3)) for row in interface_rows})
    results=[]
    for number,(upper,lower) in enumerate(zip(interfaces,interfaces[1:]),start=1):
        thickness=(lower-upper)*um_per_px
        results.append(Measurement(f"L{number} thickness (I{number}–I{number+1})",thickness,"µm","reviewed laterally averaged vertical intensity-gradient interfaces",.7,{"upper_row":upper,"lower_row":lower}))
    if len(interfaces)>=2:
        results.append(Measurement("Total stack height",(interfaces[-1]-interfaces[0])*um_per_px,"µm","outermost reviewed interfaces",.75,{"upper_row":interfaces[0],"lower_row":interfaces[-1]}))

    # A lateral repeat is only reported when the ROI spans at least three
    # correlated repeats.  This avoids inventing a pitch from a narrow stack ROI.
    lateral=cv2.GaussianBlur(gray,(5,5),0).astype(float)
    lateral=np.mean(np.abs(np.gradient(lateral,axis=1)),axis=0)
    lateral-=lateral.mean()
    ac=np.correlate(lateral,lateral,mode="full")[len(lateral)-1:]
    ac=ac/ac[0] if len(ac) and ac[0]>0 else ac
    min_lag=max(6,len(lateral)//30); max_lag=max(min_lag+1,len(lateral)//3)
    period_px=None; period_confidence=0.0
    if max_lag>min_lag+2:
        candidates=[i for i in range(min_lag+1,max_lag-1) if ac[i]>=ac[i-1] and ac[i]>=ac[i+1]]
        if candidates:
            best=max(candidates,key=lambda i:ac[i])
            if ac[best]>=.22 and len(lateral)>=3*best:
                period_px=int(best); period_confidence=float(np.clip(ac[best],0,1))
                results.append(Measurement("Lateral repeat period",period_px*um_per_px,"µm","autocorrelation of lateral edge-energy profile; at least three repeats",period_confidence,{"period_px":period_px}))
    return results,{"interface_rows":interfaces,"profile":smooth,"gradient":gradient,
                    "interface_support":[float(support[row]) for row in interfaces],
                    "interface_scores":[float(candidate_score[row]) for row in interfaces],
                    "period_px":period_px,"period_confidence":period_confidence,
                    "method":"Gaussian 7×7 vertical edges, laterally persistent transition score"}
