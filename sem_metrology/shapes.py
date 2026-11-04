from __future__ import annotations

import cv2
import numpy as np

from .roughness import low_pass_three_point, spatial_descriptors


def infer_lattice_periods(points: np.ndarray, axis: int, orthogonal_tolerance: float,
                          minimum_period: float) -> dict:
    """Infer a fundamental lattice period despite missed sites.

    Adjacent detected spacings are allowed to be integer multiples of the
    fundamental. Larger non-integer spacing families are retained as secondary
    (superlattice/block-gap) periods instead of contaminating the primary mean.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return {"fundamental_px": np.nan, "secondary_px": [], "observed_px": [], "confidence": 0.0}
    other = 1-axis; groups = []
    for idx in np.argsort(points[:, other]):
        value = points[idx, other]
        target = next((g for g in groups if abs(value-np.mean(points[g,other])) <= orthogonal_tolerance), None)
        if target is None: groups.append([idx])
        else: target.append(idx)
    gaps = []
    for group in groups:
        coords = np.sort(points[group, axis])
        gaps.extend(np.diff(coords)[np.diff(coords) > minimum_period*0.65])
    gaps = np.asarray(gaps, dtype=float)
    if not len(gaps):
        return {"fundamental_px": np.nan, "secondary_px": [], "observed_px": [], "confidence": 0.0}
    candidates = []
    for gap in gaps:
        for multiple in range(1, 7):
            candidate = gap/multiple
            if candidate >= minimum_period*0.95: candidates.append(candidate)
    best = None
    for period in candidates:
        multiples = np.maximum(1, np.rint(gaps/period))
        residual = np.abs(gaps-multiples*period)/period
        inlier = residual <= 0.12
        score = float(inlier.mean() - 0.35*np.median(np.minimum(residual,1)) + 0.002*period/minimum_period)
        if best is None or score > best[0]: best = (score, period, residual, inlier, multiples)
    _, period, residual, inlier, multiples = best
    # Refine from every inlier gap divided by its inferred missing-site count.
    fundamental_samples = gaps[inlier]/multiples[inlier]
    period = float(np.median(fundamental_samples))
    confidence = float(inlier.mean()*max(0,1-np.median(residual[inlier]) if inlier.any() else 0))
    unexplained = gaps[~inlier]
    secondary = []
    for value in np.sort(unexplained):
        cluster = next((c for c in secondary if abs(value-np.mean(c)) <= .12*np.mean(c)), None)
        if cluster is None: secondary.append([float(value)])
        else: cluster.append(float(value))
    secondary_means = [float(np.mean(c)) for c in secondary if len(c) >= 1 and np.mean(c) > 1.35*period]
    return {"fundamental_px": period, "secondary_px": sorted(secondary_means),
            "observed_px": gaps.tolist(), "confidence": confidence}


def analyze_circular_outlines(
    gray: np.ndarray,
    um_per_px: float,
    min_diameter_um: float,
    max_diameter_um: float,
    sensitivity: float = 0.88,
    exclude_border: bool = True,
    reference_roi: tuple[int, int, int, int] | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Detect circular outlines and independently refit every object's edge.

    A taught ROI is used as a normalized template for centre proposals.  The
    reported radius is never the Hough/template search radius: it is estimated
    from local radial gradient maxima and a robust least-squares circle fit.
    """
    min_radius = max(3, int(min_diameter_um / um_per_px / 2))
    max_radius = max(min_radius + 2, int(max_diameter_um / um_per_px / 2))
    work = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(work, 40, 120)
    detected = []
    if reference_roi is not None:
        x, y, rw, rh = reference_roi
        x, y = max(0, x), max(0, y)
        rw, rh = min(rw, gray.shape[1] - x), min(rh, gray.shape[0] - y)
        if rw >= 12 and rh >= 12:
            template = work[y:y + rh, x:x + rw]
            response = cv2.matchTemplate(work, template, cv2.TM_CCOEFF_NORMED)
            # Local maxima plus non-maximum suppression.  The threshold is
            # deliberately permissive; the physical edge-fit gate below is
            # responsible for accepting a metrology object.
            peak_mask = response == cv2.dilate(response, np.ones((5, 5), np.uint8))
            # Candidate generation should favor recall. The independent radial
            # edge support and residual gates below remain the metrology filter.
            threshold = max(0.24, 0.62 - 0.42 * sensitivity)
            py, px = np.where(peak_mask & (response >= threshold))
            proposals = sorted(((float(response[yy, xx]), xx + rw / 2, yy + rh / 2)
                                for yy, xx in zip(py, px)), reverse=True)
            nominal_radius = 0.25 * (rw + rh)
            for score, cx, cy in proposals:
                if any(np.hypot(cx - old[0], cy - old[1]) < 0.65 * min(rw, rh) for old in detected):
                    continue
                detected.append((cx, cy, nominal_radius))
    # Hough proposals complement the taught template so legitimate process
    # variation is not suppressed merely because an object differs from the
    # operator's reference.  The subsequent residual gate remains decisive.
    circles = cv2.HoughCircles(
        work, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(2 * min_radius, 10), param1=120,
        param2=max(8, int(55 * (1.0 - sensitivity))),
        minRadius=min_radius, maxRadius=max_radius,
    )
    hough = [] if circles is None else np.asarray(circles[0], dtype=float)
    if len(detected):
        detected.extend(tuple(row) for row in hough)
    else:
        detected = hough
    height, width = gray.shape
    scored = []
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient_gate = float(np.percentile(gradient, 70))
    for cx, cy, proposal_radius in detected:
        # Refit locally around the taught/Hough proposal. Searching the entire
        # global credible-size interval lets a profile jump to a neighbouring
        # feature when max size is intentionally broad (for example 1000 nm).
        proposal_radius=float(np.clip(proposal_radius,min_radius,max_radius))
        local_min=max(min_radius,int(np.floor(.58*proposal_radius)))
        local_max=min(max_radius,int(np.ceil(1.48*proposal_radius)))
        if local_max-local_min<3:
            local_min=max(min_radius,int(proposal_radius)-3); local_max=min(max_radius,int(proposal_radius)+3)
        radial_samples=np.arange(local_min,local_max+1,dtype=float)
        points, strengths, point_angles = [], [], []
        for angle in angles:
            xs = np.clip(np.rint(cx + radial_samples * np.cos(angle)).astype(int), 0, width - 1)
            ys = np.clip(np.rint(cy + radial_samples * np.sin(angle)).astype(int), 0, height - 1)
            profile = gradient[ys, xs]
            best = int(np.argmax(profile)); strength = float(profile[best])
            if strength >= gradient_gate:
                radius = radial_samples[best]
                points.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
                strengths.append(strength)
                point_angles.append(angle)
        support = len(points) / len(angles)
        if support < 0.38:
            continue
        pts = np.asarray(points, dtype=float)
        # Algebraic circle fit, followed by a robust residual rejection/refit.
        def fit_circle(sample):
            aa = np.c_[2 * sample[:, 0], 2 * sample[:, 1], np.ones(len(sample))]
            bb = sample[:, 0] ** 2 + sample[:, 1] ** 2
            fx, fy, c = np.linalg.lstsq(aa, bb, rcond=None)[0]
            return float(fx), float(fy), float(np.sqrt(max(c + fx * fx + fy * fy, 0)))
        fx, fy, radius = fit_circle(pts)
        signed = np.hypot(pts[:, 0] - fx, pts[:, 1] - fy) - radius; residuals = np.abs(signed)
        limit = max(1.5, 2.8 * 1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))))
        inliers = residuals <= limit
        if inliers.sum() >= 30:
            fx, fy, radius = fit_circle(pts[inliers])
            signed = np.hypot(pts[inliers, 0] - fx, pts[inliers, 1] - fy) - radius; residuals = np.abs(signed)
        else:
            signed = np.hypot(pts[:, 0] - fx, pts[:, 1] - fy) - radius
        fit_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        cer_rms=float(np.std(signed,ddof=1)) if len(signed)>1 else np.nan
        filtered=low_pass_three_point(signed); cer_filtered=float(np.std(filtered,ddof=1)) if len(filtered)>1 else np.nan
        correlation,wavelength=spatial_descriptors(signed,2*np.pi*radius/max(1,len(signed)))
        # Reject geometrically plausible aliases between neighbouring rings.
        # Their radial samples often find real edges, but cannot form one
        # coherent circle and therefore have a substantially larger residual.
        if min_radius * 0.85 <= radius <= max_radius * 1.15 and fit_rmse <= max(1.25, 0.06 * radius):
            scored.append((support, fx, fy, radius, fit_rmse,cer_rms,cer_filtered,correlation,wavelength))
    # A physical ring may generate several nearby Hough candidates. Retain only
    # the best-supported non-overlapping model so one label means one object.
    accepted = []
    for support, cx, cy, radius, fit_rmse,cer_rms,cer_filtered,correlation,wavelength in sorted(scored, reverse=True):
        touches = cx - radius <= 0 or cy - radius <= 0 or cx + radius >= width or cy + radius >= height
        if exclude_border and touches:
            continue
        if any(np.hypot(cx - old[0], cy - old[1]) < 0.90 * (radius + old[2]) for old in accepted):
            continue
        accepted.append((cx, cy, radius, support, fit_rmse,cer_rms,cer_filtered,correlation,wavelength))
    # Do not apply a single-population radius filter here. Physical size classes
    # are resolved later by the conservative class-aware population model.
    accepted.sort(key=lambda row: (row[1], row[0]))
    features = []
    centers = np.asarray([[c[0], c[1]] for c in accepted], dtype=float)
    nearest = np.full(len(accepted), np.nan)
    if len(centers) > 1:
        delta = centers[:, None, :] - centers[None, :, :]
        distances = np.sqrt((delta ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
    nominal_extent = 2*float(np.median([row[2] for row in accepted])) if accepted else 1.0
    xperiod = infer_lattice_periods(centers, 0, .55*nominal_extent, nominal_extent)
    yperiod = infer_lattice_periods(centers, 1, .55*nominal_extent, nominal_extent)
    for index, ((cx, cy, radius, support, fit_rmse,cer_rms,cer_filtered,correlation,wavelength), nn_px) in enumerate(zip(accepted, nearest), start=1):
        diameter_um = 2 * radius * um_per_px
        feature = {
            "Object": index,
            "Centroid X (px)": round(float(cx), 2),
            "Centroid Y (px)": round(float(cy), 2),
            "Diameter (µm)": diameter_um,
            "Radius (µm)": radius * um_per_px,
            "Projected area (µm²)": np.pi * (radius * um_per_px) ** 2,
            "Fitted perimeter (µm)": np.pi * diameter_um,
            "Nearest-neighbor pitch (µm)": nn_px * um_per_px,
            "Edge fit support (%)": support * 100,
            "Circle fit RMSE (µm)": fit_rmse * um_per_px,
            "Circle edge roughness RMS (µm)": cer_rms*um_per_px,
            "Circle edge roughness 3σ (µm)": 3*cer_rms*um_per_px,
            "Filtered circle edge roughness 3σ (µm)": 3*cer_filtered*um_per_px,
            "Circle edge correlation length (µm)": correlation*um_per_px,
            "Circle edge dominant wavelength (µm)": wavelength*um_per_px,
            "Circle roughness method": "radial residual from robust best-fit circle; raw plus disclosed 3-point low-pass; SEM-noise inclusive",
            "Touches border": False,
        }
        for axis, model in (("X", xperiod), ("Y", yperiod)):
            feature[f"Fundamental period {axis} (µm)"] = model["fundamental_px"]*um_per_px
            feature[f"Period {axis} confidence (%)"] = model["confidence"]*100
            for family in range(2):
                values = model["secondary_px"]
                feature[f"Secondary period {axis} #{family+1} (µm)"] = values[family]*um_per_px if family < len(values) else np.nan
        features.append(feature)
    circle_geometry = np.asarray([[row[0],row[1],row[2]] for row in accepted], dtype=float)
    return features, edges, circle_geometry


def analyze_shapes(
    gray: np.ndarray,
    um_per_px: float,
    polarity: str = "bright",
    min_area_px: int = 20,
    max_area_px: int | None = None,
    exclude_border: bool = True,
    nominal_diameter_px: float | None = None,
    sensitivity: float = 0.82,
    reference_template: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Segment irregular particles, including resolvable touching contacts.

    The binary mask defines the visible projected material.  Marker-controlled
    watershed is used only where a connected foreground island contains more
    than one credible distance-transform maximum.  It cannot reconstruct an
    occluded boundary, so contact-split objects are explicitly identified in
    the returned rows and are kept behind the application's REVIEW gate.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_mode = cv2.THRESH_BINARY if polarity == "bright" else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(blur, 0, 255, threshold_mode | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    height, width = gray.shape
    gx=cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
    edge_magnitude=cv2.magnitude(gx,gy); edge_gate=float(np.percentile(edge_magnitude,65))
    max_area_px = max_area_px or height * width
    # Estimate a useful particle scale from the smaller connected islands when
    # the operator has not supplied a credible tight reference.  The lower
    # half is deliberately used because large islands are commonly contacts.
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw_areas = sorted(float(cv2.contourArea(c)) for c in raw_contours
                       if min_area_px <= cv2.contourArea(c) <= max_area_px)
    if not nominal_diameter_px or not np.isfinite(nominal_diameter_px):
        basis = raw_areas[:max(1, (len(raw_areas) + 1)//2)]
        nominal_area = float(np.median(basis)) if basis else float(min_area_px*2)
        nominal_diameter_px = np.sqrt(4*nominal_area/np.pi)
    nominal_diameter_px = float(max(5.0, nominal_diameter_px))
    nominal_radius = nominal_diameter_px/2

    raw_component_count, raw_components = cv2.connectedComponents(mask)
    guided_watershed=False; seed_labels=np.zeros_like(mask,np.int32)
    template=(np.asarray(reference_template,dtype=np.uint8) if reference_template is not None else None)
    if template is not None and template.ndim==2 and min(template.shape)>=12 and template.shape[0]<=height and template.shape[1]<=width:
        # A taught particle supplies instance centres, not a copied contour.
        # Watershed then follows the local SEM gradient around every centre.
        response=cv2.matchTemplate(blur,cv2.GaussianBlur(template,(5,5),0),cv2.TM_CCOEFF_NORMED)
        peak_window=max(5,int(round(.34*nominal_diameter_px))); peak_window += 1-peak_window%2
        peak=response>=cv2.dilate(response,np.ones((peak_window,peak_window),np.uint8))-1e-6
        threshold=max(.20,.48-.28*float(sensitivity))
        py,px=np.where(peak & (response>=threshold))
        proposals=sorted(((float(response[yy,xx]),xx+template.shape[1]/2,yy+template.shape[0]/2)
                          for yy,xx in zip(py,px)),reverse=True)
        centres=[]
        for score,cx,cy in proposals:
            if any(np.hypot(cx-ox,cy-oy)<.40*nominal_diameter_px for ox,oy in centres): continue
            centres.append((cx,cy))
        # Bright-edge SEM contrast often leaves crescent fragments in an Otsu
        # mask. Close those fragments at the taught scale to form a conservative
        # particle envelope, then supplement template centres with envelope
        # distance maxima for particles whose shape differs from the reference.
        support_size=max(5,int(round(.30*nominal_diameter_px))); support_size += 1-support_size%2
        support=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(support_size,support_size)))
        support=cv2.dilate(support,np.ones((3,3),np.uint8))
        support_distance=cv2.distanceTransform(support,cv2.DIST_L2,5)
        support_window=max(5,int(round(.45*nominal_diameter_px))); support_window += 1-support_window%2
        support_peaks=(support_distance>=cv2.dilate(support_distance,np.ones((support_window,support_window),np.uint8))-1e-6) & (support_distance>.12*nominal_diameter_px)
        peak_count,peak_labels=cv2.connectedComponents(support_peaks.astype(np.uint8))
        for peak_id in range(1,peak_count):
            yy,xx=np.where(peak_labels==peak_id)
            if not len(xx): continue
            best=int(np.argmax(support_distance[yy,xx])); candidate=(float(xx[best]),float(yy[best]))
            if all(np.hypot(candidate[0]-ox,candidate[1]-oy)>=.38*nominal_diameter_px for ox,oy in centres): centres.append(candidate)
        if len(centres)>=2:
            guided_watershed=True; markers=np.zeros_like(gray,np.int32)
            for marker_id,(cx,cy) in enumerate(centres,start=2):
                cv2.circle(markers,(int(round(cx)),int(round(cy))),max(2,int(round(.08*nominal_diameter_px))),marker_id,-1)
            # The closed envelope is a hard support constraint. This is what
            # prevents watershed regions from leaking into noisy background.
            markers[support==0]=1
            watershed_image=cv2.cvtColor(cv2.GaussianBlur(gray,(3,3),0),cv2.COLOR_GRAY2BGR)
            cv2.watershed(watershed_image,markers); labels=markers
            label_ids=list(range(2,len(centres)+2))
            mask=np.where(labels>1,255,0).astype(np.uint8)
    if not guided_watershed:
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # One local maximum per particle core.  A fairly large NMS footprint stops
        # SEM texture from creating multiple seeds inside one irregular grain.
        footprint = max(3, int(round(.55*nominal_diameter_px)))
        if footprint % 2 == 0: footprint += 1
        local_max = distance >= cv2.dilate(distance, np.ones((footprint, footprint), np.uint8)) - 1e-6
        core_gate = max(1.5, nominal_radius*(0.20 + 0.12*(1.0-float(sensitivity))))
        seeds = (local_max & (distance >= core_gate) & (mask > 0)).astype(np.uint8)
        seed_count, seed_labels = cv2.connectedComponents(seeds)
        if seed_count > 1:
            markers = seed_labels.astype(np.int32) + 1
            markers[mask == 0] = 1; markers[(mask > 0) & (seeds == 0)] = 0
            watershed_image = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)
            cv2.watershed(watershed_image, markers); labels = markers
            label_ids = [value for value in np.unique(labels) if value > 1]
        else:
            component_count, components = cv2.connectedComponents(mask)
            labels = components + 1; label_ids = list(range(2, component_count + 1))

    features = []
    accepted_contours = []
    for label_id in label_ids:
        object_mask = np.where(labels == label_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        # Refine exposed boundaries to the strongest local SEM gradient along
        # the contour normal. Shared watershed dividers are intentionally left
        # unchanged: moving them would imply knowledge of an occluded edge.
        points=contour.reshape(-1,2).astype(np.float32); refined=points.copy(); localized=0
        if len(points)>=7:
            for point_index,point in enumerate(points):
                x0=int(np.clip(round(float(point[0])),0,width-1)); y0=int(np.clip(round(float(point[1])),0,height-1))
                neighbourhood=np.unique(labels[max(0,y0-2):min(height,y0+3),max(0,x0-2):min(width,x0+3)])
                if any(value>1 and value!=label_id for value in neighbourhood): continue
                tangent=points[(point_index+2)%len(points)]-points[(point_index-2)%len(points)]
                length=float(np.hypot(tangent[0],tangent[1]))
                if length<1e-6: continue
                normal=np.asarray([-tangent[1],tangent[0]],np.float32)/length
                offsets=np.arange(-4,5,dtype=np.float32); samples=point[None,:]+offsets[:,None]*normal[None,:]
                sx=np.clip(np.rint(samples[:,0]).astype(int),0,width-1); sy=np.clip(np.rint(samples[:,1]).astype(int),0,height-1)
                strengths=edge_magnitude[sy,sx]; best=int(np.argmax(strengths))
                if float(strengths[best])<edge_gate: continue
                offset=float(offsets[best])
                if 0<best<len(offsets)-1:
                    left,centre,right=(float(strengths[best-1]),float(strengths[best]),float(strengths[best+1]))
                    denominator=left-2*centre+right
                    if abs(denominator)>1e-6: offset+=float(np.clip(.5*(left-right)/denominator,-.5,.5))
                refined[point_index]=point+offset*normal; localized+=1
            contour=refined.reshape(-1,1,2).astype(np.float32)
        area_px = float(cv2.contourArea(contour))
        if not min_area_px <= area_px <= max_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        touches_border = x <= 0 or y <= 0 or x + w >= width or y + h >= height
        if exclude_border and touches_border:
            continue
        perimeter_px = float(cv2.arcLength(contour, True))
        if perimeter_px <= 0:
            continue
        hull = cv2.convexHull(contour)
        hull_area_px = float(cv2.contourArea(hull))
        (_, _), (feret_a_px, feret_b_px), rect_angle = cv2.minAreaRect(contour)
        feret_max_px = max(feret_a_px, feret_b_px)
        feret_min_px = min(feret_a_px, feret_b_px)
        major_px, minor_px, orientation = feret_max_px, feret_min_px, float(rect_angle)
        if len(contour) >= 5:
            (_, _), axes, orientation = cv2.fitEllipse(contour)
            major_px, minor_px = max(axes), min(axes)
        moments = cv2.moments(contour)
        cx = moments["m10"] / moments["m00"] if moments["m00"] else x + w / 2
        cy = moments["m01"] / moments["m00"] if moments["m00"] else y + h / 2
        scale2 = um_per_px ** 2
        equivalent_px = np.sqrt(4 * area_px / np.pi)
        # A watershed boundary or a strong concavity indicates that the visible
        # contour participated in a contact split.  Such dimensions describe
        # the visible segment, not a reconstructed hidden particle boundary.
        center_x=int(np.clip(round(cx),0,width-1)); center_y=int(np.clip(round(cy),0,height-1))
        if guided_watershed:
            boundary=cv2.dilate(object_mask,np.ones((3,3),np.uint8)) & np.where(labels==-1,255,0).astype(np.uint8)
            neighbours=np.unique(labels[cv2.dilate(boundary,np.ones((3,3),np.uint8))>0])
            watershed_boundary=bool(any(value>1 and value!=label_id for value in neighbours))
        else:
            source_component=int(raw_components[center_y,center_x])
            source_seed_ids=np.unique(seed_labels[raw_components == source_component]) if source_component else np.asarray([])
            source_seed_ids=source_seed_ids[source_seed_ids > 0]
            watershed_boundary = len(source_seed_ids) > 1
        solidity = area_px / hull_area_px if hull_area_px else np.nan
        contact_split = watershed_boundary or (np.isfinite(solidity) and solidity < 0.88)
        segmentation_confidence = float(np.clip(
            0.96 - (0.22 if contact_split else 0.0) - 0.35*max(0.0, 0.72-solidity), 0.25, 0.98
        ))
        features.append({
            "Object": len(features) + 1,
            "Centroid X (px)": round(cx, 2),
            "Centroid Y (px)": round(cy, 2),
            "Area (µm²)": area_px * scale2,
            "Perimeter (µm)": perimeter_px * um_per_px,
            "Equivalent diameter (µm)": equivalent_px * um_per_px,
            "Feret max (µm)": feret_max_px * um_per_px,
            "Feret min (µm)": feret_min_px * um_per_px,
            "Major axis (µm)": major_px * um_per_px,
            "Minor axis (µm)": minor_px * um_per_px,
            "Aspect ratio": major_px / minor_px if minor_px else np.nan,
            "Circularity": 4 * np.pi * area_px / (perimeter_px ** 2),
            "Solidity": solidity,
            "Orientation (°)": orientation,
            "Contact split": contact_split,
            "Segmentation confidence (%)": segmentation_confidence*100,
            "Boundary gradient support (%)": 100*localized/max(1,len(points)),
            "Boundary localization": "exposed contour normal, quadratic gradient peak; contact dividers unchanged",
            "Segmentation status": "REVIEW: visible contact split" if contact_split else "isolated visible contour",
            "Touches border": touches_border,
        })
        accepted_contours.append(contour)
    return features, mask, np.array(accepted_contours, dtype=object)


def analyze_rectangular_features(
    gray: np.ndarray,
    um_per_px: float,
    reference_roi: tuple[int, int, int, int] | None,
    sensitivity: float = 0.82,
    exclude_border: bool = True,
) -> tuple[list[dict], list[tuple[float, float, float, float]], dict]:
    """Measure a taught periodic rectangular/elongated feature population.

    Template correlation only locates objects. Each object's four edges are
    independently determined from local gradient profiles, so length/width
    dispersion and edge roughness are measured rather than copied from the ROI.
    """
    if reference_roi is None:
        return [], [], {}
    x, y, rw, rh = (int(v) for v in reference_roi)
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, gray.shape[1] - x), min(rh, gray.shape[0] - y)
    if rw < 14 or rh < 14:
        return [], [], {}
    work = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    template = work[y:y + rh, x:x + rw]
    response = cv2.matchTemplate(work, template, cv2.TM_CCOEFF_NORMED)
    peak_mask = response == cv2.dilate(response, np.ones((5, 5), np.uint8))
    threshold = max(0.50, 0.78 - 0.30 * sensitivity)
    py, px = np.where(peak_mask & (response >= threshold))
    proposals = sorted(((float(response[yy, xx]), xx + rw/2, yy + rh/2)
                        for yy, xx in zip(py, px)), reverse=True)
    centers = []
    for score, cx, cy in proposals:
        if any(abs(cx-ox) < 0.62*rw and abs(cy-oy) < 0.62*rh for ox, oy, _ in centers):
            continue
        if exclude_border and (cx-rw/2 <= 0 or cy-rh/2 <= 0 or cx+rw/2 >= gray.shape[1] or cy+rh/2 >= gray.shape[0]):
            continue
        centers.append((cx, cy, score))

    features, boxes = [], []
    measured_centers = []
    for cx, cy, score in centers:
        left0, top0 = int(round(cx-rw/2)), int(round(cy-rh/2))
        right0, bottom0 = left0+rw, top0+rh
        if left0 < 0 or top0 < 0 or right0 > gray.shape[1] or bottom0 > gray.shape[0]:
            continue
        local_gx = np.abs(gx[top0:bottom0, left0:right0])
        local_gy = np.abs(gy[top0:bottom0, left0:right0])
        col = local_gx.mean(axis=0); row = local_gy.mean(axis=1)
        # Search each edge in its corresponding outer half; this remains
        # stable for rounded corners and textured interiors.
        lrange = slice(max(0, int(.05*rw)), max(2, int(.48*rw)))
        rstart = min(rw-2, int(.52*rw)); rrange = slice(rstart, max(rstart+1, int(.95*rw)))
        trange = slice(max(0, int(.05*rh)), max(2, int(.48*rh)))
        bstart = min(rh-2, int(.52*rh)); brange = slice(bstart, max(bstart+1, int(.95*rh)))
        left = lrange.start + int(np.argmax(col[lrange])); right = rrange.start + int(np.argmax(col[rrange]))
        top = trange.start + int(np.argmax(row[trange])); bottom = brange.start + int(np.argmax(row[brange]))
        if right-left < .25*rw or bottom-top < .20*rh:
            continue
        # Multi-row gauges produce local CD and line-edge roughness estimates.
        ys = range(max(top+2, 1), min(bottom-1, rh-1))
        left_trace, right_trace = [], []
        window = max(2, int(.12*rw))
        for yy in ys:
            la, lb = max(0,left-window), min(rw,left+window+1)
            ra, rb = max(0,right-window), min(rw,right+window+1)
            left_trace.append(la + int(np.argmax(local_gx[yy,la:lb])))
            right_trace.append(ra + int(np.argmax(local_gx[yy,ra:rb])))
        widths = np.asarray(right_trace, float) - np.asarray(left_trace, float)
        xs = range(max(left+2, 1), min(right-1, rw-1))
        top_trace, bottom_trace = [], []
        vwindow = max(2, int(.14*rh))
        for xx in xs:
            ta, tb = max(0,top-vwindow), min(rh,top+vwindow+1)
            ba, bb = max(0,bottom-vwindow), min(rh,bottom+vwindow+1)
            top_trace.append(ta + int(np.argmax(local_gy[ta:tb,xx])))
            bottom_trace.append(ba + int(np.argmax(local_gy[ba:bb,xx])))
        heights = np.asarray(bottom_trace,float)-np.asarray(top_trace,float)
        length_px = float(np.median(widths)) if len(widths) else float(right-left)
        width_px = float(np.median(heights)) if len(heights) else float(bottom-top)
        edge_sigmas = [np.std(trace,ddof=1) for trace in (left_trace,right_trace,top_trace,bottom_trace) if len(trace)>1]
        ler_px = float(np.mean(edge_sigmas)) if edge_sigmas else 0.0
        global_box = (left0+left, top0+top, left0+right, top0+bottom)
        geometry={"box":global_box,"edges":[
            [(left0+xx,top0+yy) for xx,yy in zip(left_trace,ys)],
            [(left0+xx,top0+yy) for xx,yy in zip(right_trace,ys)],
            [(left0+xx,top0+yy) for xx,yy in zip(xs,top_trace)],
            [(left0+xx,top0+yy) for xx,yy in zip(xs,bottom_trace)],
        ]}
        boxes.append(geometry); measured_centers.append(((global_box[0]+global_box[2])/2, (global_box[1]+global_box[3])/2))
        features.append({
            "Object": len(features)+1, "Centroid X (px)": cx, "Centroid Y (px)": cy,
            "Length (µm)": length_px*um_per_px, "Mean width (µm)": width_px*um_per_px,
            "Length 3σ (µm)": (3*np.std(widths,ddof=1)*um_per_px if len(widths)>1 else 0.0),
            "Width 3σ (µm)": (3*np.std(heights,ddof=1)*um_per_px if len(heights)>1 else 0.0),
            "LER 3σ (µm)": 3*ler_px*um_per_px, "Aspect ratio": length_px/max(width_px,1e-9),
            "Template match (%)": score*100, "Period X (µm)": np.nan, "Period Y (µm)": np.nan,
            "Touches border": False,
        })
    points = np.asarray(measured_centers, float)
    xmodel = infer_lattice_periods(points, 0, .55*rh, rw)
    ymodel = infer_lattice_periods(points, 1, .55*rw, rh)
    px, py = xmodel["fundamental_px"], ymodel["fundamental_px"]
    sx = xmodel["secondary_px"][0] if xmodel["secondary_px"] else np.nan
    sy = ymodel["secondary_px"][0] if ymodel["secondary_px"] else np.nan
    for row_data in features:
        row_data["Period X (µm)"] = px*um_per_px
        row_data["Period Y (µm)"] = py*um_per_px
        row_data["Secondary period X (µm)"] = sx*um_per_px
        row_data["Secondary period Y (µm)"] = sy*um_per_px
    diagnostics = {"period_x": xmodel, "period_y": ymodel}
    return features, boxes, diagnostics


def population_summary(features: list[dict]) -> list[dict]:
    if not features:
        return []
    excluded = {"Object", "Centroid X (px)", "Centroid Y (px)", "Touches border",
                "Contact split", "Segmentation status"}
    rows = []
    for metric in features[0]:
        if metric in excluded:
            continue
        try:
            values = np.asarray([row[metric] for row in features], dtype=float)
        except (TypeError, ValueError):
            continue
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        mean = float(values.mean())
        rows.append({
            "Metric": metric,
            "N": len(values),
            "Mean": mean,
            "Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "3σ": float(3 * values.std(ddof=1)) if len(values) > 1 else np.nan,
            "Median": float(np.median(values)),
            "Min": float(values.min()),
            "Max": float(values.max()),
            "Range": float(values.max() - values.min()),
            "CV (%)": float(values.std(ddof=1) / abs(mean) * 100) if len(values) > 1 and mean else 0.0,
        })
    return rows
