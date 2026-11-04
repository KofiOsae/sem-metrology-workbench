"""Generate redistributable SEM-like tutorial images with known geometry.

These are algorithm/training fixtures, not physical SEM acquisitions. Randomness
is deterministic. The generated images and ground_truth.json are project
material under the repository BSD-3-Clause license.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent/"generated"
RNG=np.random.default_rng(20261104)
NM_PER_PX=5.0


def sem_background(height=600,width=900,mean=92,noise=8):
    image=RNG.normal(mean,noise,(height,width)).astype(np.float32)
    image+=np.linspace(-5,5,width,dtype=np.float32)[None,:]
    return image


def finish(gray,name,bar_nm=1000):
    gray=cv2.GaussianBlur(gray,(3,3),.7); gray=np.clip(gray,0,255).astype(np.uint8)
    h,w=gray.shape; footer=90; canvas=np.full((h+footer,w),28,np.uint8); canvas[:h]=gray
    bar_px=round(bar_nm/NM_PER_PX); x2=w-35; x1=x2-bar_px; y=h+48
    cv2.line(canvas,(x1,y),(x2,y),235,4); cv2.line(canvas,(x1,y-7),(x1,y+7),235,2); cv2.line(canvas,(x2,y-7),(x2,y+7),235,2)
    cv2.putText(canvas,f"{bar_nm/1000:g} um",(x1,y-14),cv2.FONT_HERSHEY_SIMPLEX,.62,235,1,cv2.LINE_AA)
    cv2.putText(canvas,"SYNTHETIC TRAINING IMAGE - 5.000 nm/px",(20,h+76),cv2.FONT_HERSHEY_SIMPLEX,.46,190,1,cv2.LINE_AA)
    Image.fromarray(canvas).save(OUT/name)


def circles():
    image=sem_background(); centers=[]
    for y in range(85,530,110):
        for x in range(90,840,125):
            radius=24 if (x//125+y//110)%2 else 18; centers.append([x,y,radius])
            # A single dark/bright transition defines the physical boundary.
            cv2.circle(image,(x,y),radius,42,-1)
    finish(image,"01_two_class_hole_array.png")
    return {"nm_per_pixel":NM_PER_PX,"objects":len(centers),"diameters_nm":[180,240],"period_x_nm":625,"period_y_nm":550}


def grating():
    image=sem_background(mean=78,noise=6)
    for x in range(60,900,95):
        edge=RNG.normal(0,1.2,600).astype(np.int32); points=np.array([[x+int(e),y] for y,e in enumerate(edge)],np.int32)
        cv2.polylines(image,[points],False,205,32); cv2.polylines(image,[points],False,225,3)
    finish(image,"02_vertical_grating.png")
    return {"nm_per_pixel":NM_PER_PX,"nominal_cd_nm":160,"pitch_nm":475,"orientation":"vertical"}


def particles():
    image=sem_background(mean=65,noise=9); count=0
    centers=[]; index=0
    for y in (120,300,480):
        for x in (120,330,540,750):
            centers.append((x,y,35+(index%4)*5,28+(index%3)*6,(index*17)%60-30)); index+=1
    for x,y,a,b,angle in centers:
        cv2.ellipse(image,(x,y),(a,b),angle,0,360,190,-1); count+=1
    finish(image,"03_irregular_particles.png")
    return {"nm_per_pixel":NM_PER_PX,"objects":count,"note":"isolated ellipses with known visible boundaries"}


def cross_section():
    image=sem_background(mean=58,noise=5); tops=[]
    cv2.rectangle(image,(0,480),(899,599),130,-1)
    for x,width,height in [(120,95,260),(310,105,275),(510,90,250),(700,110,285)]:
        top=480-height; tops.append([x,width,height]); polygon=np.array([[x,480],[x+8,top],[x+width-8,top],[x+width,480]],np.int32)
        cv2.fillPoly(image,[polygon],205)
    finish(image,"04_cross_section_pillars.png")
    return {"nm_per_pixel":NM_PER_PX,"structures":len(tops),"heights_nm":[v[2]*NM_PER_PX for v in tops]}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    truth={"license":"BSD-3-Clause","purpose":"synthetic tutorial/regression data; not physical SEM", "samples":{
        "01_two_class_hole_array.png":circles(),"02_vertical_grating.png":grating(),
        "03_irregular_particles.png":particles(),"04_cross_section_pillars.png":cross_section()}}
    (OUT/"ground_truth.json").write_text(json.dumps(truth,indent=2),encoding="utf-8")
    print(f"Generated {len(truth['samples'])} samples in {OUT}")


if __name__=="__main__": main()
