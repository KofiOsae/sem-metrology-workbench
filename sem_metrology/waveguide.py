from __future__ import annotations

import heapq
import cv2
import numpy as np


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning; preserves connected curved centerlines."""
    image=(mask>0).astype(np.uint8); changed=True
    while changed:
        changed=False
        for first in (True,False):
            p=np.pad(image,1); p2=p[:-2,1:-1]; p3=p[:-2,2:]; p4=p[1:-1,2:]; p5=p[2:,2:]; p6=p[2:,1:-1]; p7=p[2:,:-2]; p8=p[1:-1,:-2]; p9=p[:-2,:-2]
            neighbours=p2+p3+p4+p5+p6+p7+p8+p9
            transitions=sum(condition.astype(np.uint8) for condition in (((p2==0)&(p3==1)),((p3==0)&(p4==1)),((p4==0)&(p5==1)),((p5==0)&(p6==1)),((p6==0)&(p7==1)),((p7==0)&(p8==1)),((p8==0)&(p9==1)),((p9==0)&(p2==1))))
            if first: constraint=(p2*p4*p6==0)&(p4*p6*p8==0)
            else: constraint=(p2*p4*p8==0)&(p2*p6*p8==0)
            remove=(image==1)&(neighbours>=2)&(neighbours<=6)&(transitions==1)&constraint
            if np.any(remove): image[remove]=0; changed=True
    return image.astype(bool)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
    if count<=1: return np.zeros_like(binary,dtype=bool)
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])); return labels==index


def _graph_path(skeleton: np.ndarray) -> list[tuple[int,int]]:
    points=np.argwhere(skeleton); nodes={tuple(p):i for i,p in enumerate(points)}
    if len(points)<2: return [tuple(p) for p in points]
    neighbours=[[] for _ in points]
    for i,(y,x) in enumerate(points):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dx or dy): continue
                j=nodes.get((y+dy,x+dx))
                if j is not None: neighbours[i].append((j,float(np.hypot(dx,dy))))
    def farthest(start):
        distance=np.full(len(points),np.inf); parent=np.full(len(points),-1,int); distance[start]=0; queue=[(0.0,start)]
        while queue:
            value,node=heapq.heappop(queue)
            if value!=distance[node]: continue
            for other,weight in neighbours[node]:
                proposal=value+weight
                if proposal<distance[other]: distance[other]=proposal; parent[other]=node; heapq.heappush(queue,(proposal,other))
        reachable=np.flatnonzero(np.isfinite(distance)); end=int(reachable[np.argmax(distance[reachable])]); return end,distance,parent
    endpoints=[i for i,n in enumerate(neighbours) if len(n)==1]; start=endpoints[0] if endpoints else 0
    a,_,_=farthest(start); b,_,parent=farthest(a); order=[]; node=b
    while node>=0: order.append(node); node=int(parent[node])
    return [tuple(points[i]) for i in order[::-1]]


def analyze_waveguide(gray: np.ndarray, unit_per_px: float, region_mask: np.ndarray | None=None) -> tuple[list[dict],dict]:
    """Measure one filled, non-branching waveguide inside an operator-reviewed region."""
    blur=cv2.GaussianBlur(gray,(5,5),0); threshold,_=cv2.threshold(blur,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    allowed=np.ones_like(gray,dtype=bool) if region_mask is None else region_mask.astype(bool)
    candidates=[]
    for foreground in (blur>threshold,blur<=threshold):
        component=_largest_component(foreground & allowed); area=int(component.sum())
        if area<25 or area>0.65*max(1,int(allowed.sum())): continue
        contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        perimeter=max((cv2.arcLength(c,True) for c in contours),default=0); candidates.append((area/max(perimeter,1),component,contours))
    if not candidates: return [],{}
    _,component,contours=max(candidates,key=lambda item:item[0]); distance=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
    skeleton=_skeletonize(component); path=_graph_path(skeleton)
    if len(path)<5: return [],{}
    cumulative=[0.0]
    for (y0,x0),(y1,x1) in zip(path,path[1:]): cumulative.append(cumulative[-1]+float(np.hypot(x1-x0,y1-y0)))
    cumulative=np.asarray(cumulative); widths=np.asarray([2*distance[y,x] for y,x in path]); keep=np.linspace(0,len(path)-1,min(300,len(path))).astype(int); keep=np.unique(keep)
    path_length=float(cumulative[-1]*unit_per_px); position=cumulative[keep]*unit_per_px; local_width=widths[keep]*unit_per_px
    slope=float(np.polyfit(position,local_width,1)[0]) if len(position)>1 and np.ptp(position)>0 else np.nan
    rows=[{"Gauge":i+1,"Centerline X (px)":float(path[k][1]),"Centerline Y (px)":float(path[k][0]),"Arc position (µm)":float(position[i]),"Normal width (µm)":float(local_width[i])} for i,k in enumerate(keep)]
    diagnostics={"mask":component,"contours":contours,"path_xy":np.asarray([(x,y) for y,x in path],np.int32),"path_length":path_length,"mean_width":float(local_width.mean()),"std_width":float(local_width.std(ddof=1)) if len(local_width)>1 else np.nan,"min_width":float(local_width.min()),"max_width":float(local_width.max()),"taper":slope,"branch_pixels":int(sum(1 for y,x in path if np.count_nonzero(skeleton[max(0,y-1):y+2,max(0,x-1):x+2])>4))}
    return rows,diagnostics


def waveguide_overlay(rgb: np.ndarray, diagnostics: dict) -> np.ndarray:
    canvas=rgb.copy(); contours=[np.asarray(c,np.int32) for c in diagnostics.get("contours",[])]
    if contours: cv2.drawContours(canvas,contours,-1,(255,0,255),4)
    path=diagnostics.get("path_xy")
    if path is not None and len(path)>1: cv2.polylines(canvas,[path],False,(0,255,255),3,cv2.LINE_AA)
    return canvas
