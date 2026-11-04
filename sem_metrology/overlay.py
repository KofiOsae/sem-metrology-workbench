from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
import cv2


def calibration_overlay(rgb: np.ndarray, detection: dict | None) -> Image.Image:
    image = Image.fromarray(rgb.copy())
    if detection:
        draw = ImageDraw.Draw(image)
        draw.line([detection["start_xy"], detection["end_xy"]], fill=(255, 0, 255), width=3)
    return image


def cross_section_overlay(rgb: np.ndarray, diagnostics: dict) -> Image.Image:
    canvas=rgb.copy()
    for trace in diagnostics.get("sidewall_traces",()):
        if len(trace)>1: cv2.polylines(canvas,[np.rint(trace).astype(np.int32)],False,(255,0,255),3,cv2.LINE_AA)
    interfaces=list(diagnostics.get("interface_rows",[]))
    for index,y in enumerate(interfaces,start=1):
        cv2.line(canvas,(0,int(y)),(canvas.shape[1]-1,int(y)),(0,220,255),2,cv2.LINE_AA)
        cv2.putText(canvas,f"I{index}",(5,max(14,int(y)-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    if diagnostics.get("height_line"):
        x1,y1,x2,y2=diagnostics["height_line"]; draw.line([(x1,y1),(x2,y2)],fill=(0,255,80),width=3); draw.ellipse((x1-4,y1-4,x1+4,y1+4),fill=(0,255,80)); draw.ellipse((x2-4,y2-4,x2+4,y2+4),fill=(0,255,80)); draw.text((x1+6,(y1+y2)//2),"HEIGHT",fill=(0,255,80))
    for left, right, y in diagnostics.get("edge_points", []):
        draw.line([(left, y), (right, y)], fill=(255, 0, 255), width=2)
        draw.ellipse((left - 3, y - 3, left + 3, y + 3), fill=(0, 255, 255))
        draw.ellipse((right - 3, y - 3, right + 3, y + 3), fill=(0, 255, 255))
    if len(interfaces)>=2:
        gauge_x=max(12,canvas.shape[1]-18)
        for number,(upper,lower) in enumerate(zip(interfaces,interfaces[1:]),start=1):
            draw.line([(gauge_x,upper),(gauge_x,lower)],fill=(255,255,255),width=2)
            draw.line([(gauge_x-4,upper),(gauge_x+4,upper)],fill=(255,255,255),width=2)
            draw.line([(gauge_x-4,lower),(gauge_x+4,lower)],fill=(255,255,255),width=2)
            unit_scale=diagnostics.get("unit_scale")
            text=f"L{number}"
            if unit_scale: text+=f" {abs(lower-upper)*unit_scale:.1f} {diagnostics.get('unit','nm')}"
            draw.text((max(2,gauge_x-78),(upper+lower)//2-7),text,fill=(255,255,255))
        left=max(12,canvas.shape[1]-34)
        draw.line([(left,interfaces[0]),(left,interfaces[-1])],fill=(70,255,100),width=3)
        draw.text((max(2,left-92),max(2,(interfaces[0]+interfaces[-1])//2-8)),"TOTAL",fill=(70,255,100))
    period_px=diagnostics.get("period_px")
    if period_px:
        y=max(18,min(canvas.shape[0]-12,interfaces[0]-14 if interfaces else 18)); x1=max(8,(canvas.shape[1]-period_px)//2); x2=x1+period_px
        draw.line([(x1,y),(x2,y)],fill=(255,255,255),width=2)
        draw.line([(x1,y-4),(x1,y+4)],fill=(255,255,255),width=2); draw.line([(x2,y-4),(x2,y+4)],fill=(255,255,255),width=2)
        unit_scale=diagnostics.get("unit_scale"); label="PERIOD"
        if unit_scale: label+=f" {period_px*unit_scale:.1f} {diagnostics.get('unit','nm')}"
        draw.text((x1,max(1,y-14)),label,fill=(255,255,255))
    return image


def line_overlay(rgb: np.ndarray, lines: list[dict]) -> Image.Image:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines, start=1):
        try:
            start = (float(line["x1"]), float(line["y1"]))
            end = (float(line["x2"]), float(line["y2"]))
        except (KeyError, TypeError, ValueError):
            continue
        draw.line([start, end], fill=(255, 0, 255), width=3)
        draw.ellipse((start[0]-4, start[1]-4, start[0]+4, start[1]+4), fill=(0, 255, 255))
        draw.ellipse((end[0]-4, end[1]-4, end[0]+4, end[1]+4), fill=(0, 255, 255))
        draw.text((start[0] + 6, start[1] + 3), str(index), fill=(255, 255, 0))
    return image


def shape_overlay(rgb: np.ndarray, contours: np.ndarray, features: list[dict]) -> Image.Image:
    canvas = rgb.copy()
    for contour, row in zip(contours, features):
        colour=(255,165,0) if row.get("Contact split") else (255,0,255)
        cv2.drawContours(canvas,[np.asarray(contour,dtype=np.int32)],-1,colour,4)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for row in features:
        x, y = row["Centroid X (px)"], row["Centroid Y (px)"]
        draw.ellipse((x-3, y-3, x+3, y+3), fill=(0, 255, 255))
        label=str(row["Object"])+("•C" if row.get("Contact split") else "")
        draw.text((x+5, y-7),label,fill=(255,255,255) if row.get("Contact split") else (255,255,0))
    return image


def circle_overlay(rgb: np.ndarray, circles: np.ndarray, features: list[dict]) -> Image.Image:
    canvas = rgb.copy()
    for index, (cx, cy, radius) in enumerate(circles, start=1):
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(canvas, center, int(round(radius)), (255, 0, 255), 4)
        cv2.circle(canvas, center, 3, (0, 255, 255), -1)
        cv2.putText(canvas, str(index), (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    if len(circles)>1:
        centers=np.asarray(circles)[:,:2]; delta=centers[:,None,:]-centers[None,:,:]; dist=np.sqrt((delta**2).sum(2)); np.fill_diagonal(dist,np.inf)
        i,j=np.unravel_index(np.argmin(np.abs(dist-np.median(np.min(dist,axis=1)))),dist.shape)
        a,b=tuple(np.rint(centers[i]).astype(int)),tuple(np.rint(centers[j]).astype(int)); cv2.line(canvas,a,b,(0,255,80),2,cv2.LINE_AA)
        pitch=dist[i,j]; label=f"representative pitch {pitch:.1f} px"
        if features and "Nearest-neighbor pitch (µm)" in features[i]: label=f"representative pitch {features[i]['Nearest-neighbor pitch (µm)']*1000:.1f} nm"
        mid=((a[0]+b[0])//2,(a[1]+b[1])//2-7); cv2.putText(canvas,label,mid,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,80),1,cv2.LINE_AA)
    return Image.fromarray(canvas)


def rectangular_overlay(rgb: np.ndarray, boxes: list, features: list[dict]) -> Image.Image:
    """Show pixel-resolved edge traces rather than idealized rectangles."""
    canvas = rgb.copy()
    for row, geometry in zip(features, boxes):
        if isinstance(geometry,dict):
            left,top,right,bottom=geometry["box"]
            for edge in geometry.get("edges",[]):
                if len(edge)>1: cv2.polylines(canvas,[np.rint(edge).astype(np.int32)],False,(255,0,255),2,cv2.LINE_AA)
        else: left,top,right,bottom=geometry
        p1, p2 = (int(round(left)), int(round(top))), (int(round(right)), int(round(bottom)))
        cy = int(round((top+bottom)/2)); cx = int(round((left+right)/2))
        cv2.line(canvas, (p1[0], cy), (p2[0], cy), (0, 255, 255), 1)
        cv2.line(canvas, (cx, p1[1]), (cx, p2[1]), (0, 220, 255), 1)
        cv2.putText(canvas, str(row["Object"]), (p1[0]+3, p1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255,255,0), 1, cv2.LINE_AA)
    return Image.fromarray(canvas)


def line_space_overlay(rgb: np.ndarray, bands: list, orientation: str = "vertical") -> Image.Image:
    canvas = rgb.copy()
    height = canvas.shape[0]
    for index, band in enumerate(bands, start=1):
        if isinstance(band, dict):
            left, right = band["left"], band["right"]
            left_trace = np.asarray(band.get("left_trace", []), dtype=float)
            right_trace = np.asarray(band.get("right_trace", []), dtype=float)
            samples = np.arange(min(len(left_trace), len(right_trace)), dtype=np.int32)
            if len(samples) > 1:
                if orientation == "horizontal":
                    lp = np.column_stack((samples, np.rint(left_trace[:len(samples)]))).astype(np.int32)
                    rp = np.column_stack((samples, np.rint(right_trace[:len(samples)]))).astype(np.int32)
                else:
                    lp = np.column_stack((np.rint(left_trace[:len(samples)]), samples)).astype(np.int32)
                    rp = np.column_stack((np.rint(right_trace[:len(samples)]), samples)).astype(np.int32)
                cv2.polylines(canvas, [lp], False, (0,255,255), 2, cv2.LINE_AA)
                cv2.polylines(canvas, [rp], False, (255,0,255), 2, cv2.LINE_AA)
        else:
            left, right = band
        if orientation == "horizontal":
            if not isinstance(band, dict):
                cv2.line(canvas, (0, left), (canvas.shape[1]-1, left), (0,255,255), 2)
                cv2.line(canvas, (0, right), (canvas.shape[1]-1, right), (255,0,255), 2)
            cv2.putText(canvas, str(index), (4,left+13), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,0), 1, cv2.LINE_AA)
        else:
            if not isinstance(band, dict):
                cv2.line(canvas, (left, 0), (left, height - 1), (0, 255, 255), 2)
                cv2.line(canvas, (right, 0), (right, height - 1), (255, 0, 255), 2)
            cv2.putText(canvas, str(index), (left + 3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 0), 1, cv2.LINE_AA)
    if len(bands) > 1:
        def centre(b): return (b["left"]+b["right"])/2 if isinstance(b,dict) else (b[0]+b[1])/2
        centres=[centre(b) for b in bands]; k=max(0,(len(centres)-2)//2); a,b=centres[k],centres[k+1]
        if orientation == "horizontal": p1,p2=(20,int(round(a))),(20,int(round(b)))
        else: p1,p2=(int(round(a)),20),(int(round(b)),20)
        cv2.line(canvas,p1,p2,(0,255,80),2,cv2.LINE_AA)
        cv2.putText(canvas,f"pitch {abs(b-a):.1f} px",((p1[0]+p2[0])//2+4,(p1[1]+p2[1])//2-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,80),1,cv2.LINE_AA)
    return Image.fromarray(canvas)


def oriented_line_space_overlay(rgb: np.ndarray, bands: list, inverse_affine: np.ndarray, direction_deg: float) -> Image.Image:
    """Map pixel-resolved traces measured in a deskewed frame back to the source image."""
    canvas=rgb.copy()
    for index,band in enumerate(bands,start=1):
        for key,color in (("left_trace",(0,255,255)),("right_trace",(255,0,255))):
            trace=np.asarray(band.get(key,[]),float)
            if len(trace)<2: continue
            points=np.column_stack((trace,np.arange(len(trace),dtype=float))).reshape((-1,1,2)).astype(np.float32)
            mapped=cv2.transform(points,inverse_affine).reshape((-1,2))
            inside=(mapped[:,0]>=0)&(mapped[:,0]<canvas.shape[1])&(mapped[:,1]>=0)&(mapped[:,1]<canvas.shape[0])
            mapped=mapped[inside]
            if len(mapped)>1: cv2.polylines(canvas,[np.rint(mapped).astype(np.int32)],False,color,2,cv2.LINE_AA)
    cv2.putText(canvas,f"line direction {direction_deg:.1f} deg",(8,42),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,80),1,cv2.LINE_AA)
    return Image.fromarray(canvas)
