# Practical tutorials

## Tutorial A — circular hole array

1. Open a raw top-view image and verify scale endpoints/value.
2. Choose **Circular outlines / holes**.
3. Draw a reference tightly around one complete central hole.
4. Set a credible size interval bracketing every intended hole class.
5. Run. Review contour closure, border rejection and representative pitch.
6. Inspect diameter and fundamental X/Y period. If two diameter populations are
   proposed, inspect overlay classes and confirm them before release.
7. Plot diameter, pitch and edge roughness; hover suspected outliers.

## Tutorial B — irregular touching particles

1. Choose **Irregular particles / grains** and correct bright/dark polarity.
2. Teach one complete isolated particle, not background or an agglomerate.
3. Mask labels/scale bar if they overlap the specimen area.
4. Run. Magenta isolated contours are eligible for isolated statistics; orange
   watershed contacts are visible-segment estimates and remain REVIEW.
5. Report isolated-particle statistics as the primary population. Report the
   combined visible-segment result only with its REVIEW qualification.

## Tutorial C — line-space grating

1. Choose **Line-space / grating** and set orientation if Auto is wrong.
2. Run across the full specimen area or a representative ROI.
3. Verify edges follow pixel-resolved lines. Inspect CD, pitch, duty cycle,
   LER/LWR and period direction. Do not report finite line length for a grating
   that continues outside the frame.

## Tutorial D — cross-section pillars

1. Choose **Cross-section feature ROI**.
2. Draw a narrow ROI containing the complete top, both sidewalls and local
   substrate on each side.
3. Run and verify the green height gauge reaches the physical top and local
   substrate interface—not a cap layer or footer.
4. Repeat ROIs for at least five structures where possible. Compare height,
   CDs, angle, mean, standard deviation and distribution.

## Tutorial E — multilayer stack

1. Choose **Layer stack / film cross-section ROI** and draw through a laterally
   representative part of the stack.
2. Run, correlate I1/I2… labels with table layers L1/L2…, and delete false cyan
   interfaces. Restore candidates if needed.
3. Enter known periods spanned only when justified by the design/ROI.
4. Report reviewed layer thickness and total stack height. Contrast edges are
   candidate interfaces and should be corroborated by material knowledge.

## Tutorial F — dose/process comparison

1. Keep recipe, calibration practice and acquisition conditions consistent.
2. Analyze image 1, review it, and choose **Add reviewed result to process
   study**. Enter dose from the process record.
3. Repeat for all doses and replicate images.
4. Open the study, select a common metric/class, inspect within-image error bars,
   slope and R², and export long-form CSV for formal modeling.
