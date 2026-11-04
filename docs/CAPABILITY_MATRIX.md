# Capability matrix

| Recipe/tool | Primary quantities | Operator input | Important limitation |
|---|---|---|---|
| Circular outlines / holes | diameter, area, perimeter, pitch X/Y, circular roughness, size classes | calibration; optional reference feature | incomplete/overlapping rings require review |
| Rectangular periodic array | length, mean width, width variation, period X/Y, LER | optional reference and expected period | only design-relevant axes are reported |
| Irregular particles / grains | equivalent diameter, Feret max/min, area, perimeter, circularity, solidity | polarity; tight isolated reference recommended | contact splits describe visible segments, not hidden overlap |
| Line-space / grating | CD, pitch, duty cycle, LER/LWR, direction | line orientation/polarity when Auto is uncertain | intensity edge depends on imaging conditions |
| Single feature / contour ROI | area, perimeter, Feret axes, fitted axes, orientation | one bounded feature | generic descriptors are not automatically design dimensions |
| Curved/tapered waveguide | medial path length, normal width and taper | bounded non-branching ROI | junctions/branches are not a released model |
| Cross-section feature ROI | height, top/middle/bottom CD, sidewall angle | one ROI per structure | height uses detected local substrate datum |
| Layer stack ROI | reviewed interface positions, layer thickness, total stack height, lateral period | ROI and interface review | image contrast interfaces are candidates, not material proof |
| Line gauge/profile | drawn length, lateral chords, gaps, center spacing, intensity profile | line through features | SEM intensity is not height; off-center chord is below diameter |
| Surface texture | line Ra/Rq and areal Sa/Sq/Ssk/Sku of SEM signal | line/rectangle/freehand region | amplitude is intensity %, not nm roughness |
| Process study | response versus dose/focus/time/etc., within-image dispersion, slope/R² | trusted factor values from process record | descriptive comparison, not causal proof |

Every calibrated result uses original-image pixels, never display pixels.
