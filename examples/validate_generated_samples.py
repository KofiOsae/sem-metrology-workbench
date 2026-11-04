"""Run representative recipe checks against generated tutorial samples."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from sem_metrology.calibration import detect_footer,detect_scale_bar
from sem_metrology.lines import analyze_line_space
from sem_metrology.measurements import measure_cross_section
from sem_metrology.shapes import analyze_circular_outlines,analyze_shapes

ROOT=Path(__file__).resolve().parent/"generated"


def gray(name): return np.asarray(Image.open(ROOT/name).convert("L"))[:600]


def main():
    for path in ROOT.glob("*.png"):
        rgb=np.asarray(Image.open(path).convert("RGB")); footer=detect_footer(rgb); bar=detect_scale_bar(rgb,footer)
        assert footer==600,(path.name,footer)
        assert bar and abs(bar["pixel_length"]-200)<=2,(path.name,bar)

    rows,_,_=analyze_circular_outlines(gray("01_two_class_hole_array.png"),.005,.12,.30,.90,True,None)
    assert len(rows)==30,len(rows)
    diameters=np.asarray([row["Diameter (µm)"]*1000 for row in rows])
    assert np.count_nonzero(diameters<205)==15 and np.count_nonzero(diameters>=205)==15,diameters

    rows,_,_=analyze_line_space(gray("02_vertical_grating.png"),.005,"bright",.10,.25,"vertical")
    assert len(rows)==9,len(rows)
    assert 165<=np.mean([row["Mean CD (µm)"]*1000 for row in rows])<=190

    rows,_,_=analyze_shapes(gray("03_irregular_particles.png"),.005,"bright",500,20000,True)
    assert len(rows)==14,len(rows)

    measurements,diagnostics=measure_cross_section(gray("04_cross_section_pillars.png")[:,100:230],.005)
    height_nm=next(m.value*1000 for m in measurements if m.name=="Feature height")
    assert 1270<=height_nm<=1340,height_nm
    assert diagnostics.get("height_datum")=="feature top to local substrate interface"
    print("PASS: calibration, two-class circles, grating, irregular particles and cross-section datum")


if __name__=="__main__": main()
