# User guide — Version 1.1.0

## Workspace layout

The workbench uses three adjustable columns. **Settings** on the left contains
calibration, recipe, detection and region controls; collapse a section when it
is not needed. The **centre** keeps the calibrated image and overlay above a
dropdown-selectable measurements, statistics, distribution or QA view.
**Diagnostics** on the right contains the run control, release state and model
warnings. Drag either vertical divider or the image/results divider to suit the
screen; the workbench remembers these proportions between sessions.

## 1. Before measuring

Use the original SEM acquisition where possible. Avoid screenshots, JPEG
recompression, burned-in annotations across the specimen, saturation, charging,
strong drift and insufficient sampling. Preserve the raw image and acquisition
metadata. The footer is displayed for traceability but excluded from analysis.

## 2. Calibrate

Calibration priority is: trusted companion `PixelSize` metadata, then a reviewed
scale bar, then original-image pixels when explicitly enabled.

The magenta scale overlay is only an automatic proposal. Confirm both endpoint
dots coincide with the physical bar. If not, enter the printed value and unit,
press **Redraw scale-bar endpoints**, draw exactly from one endpoint to the
other, then check **Calibration source reviewed and confirmed**. Do not include
the label text or unrelated ruler ticks.

For quantitative work, verify SEM magnification against an appropriate reference
material and include calibration, repeatability, sampling, edge-model and
operator contributions in the laboratory uncertainty budget.

## 3. Choose the physical recipe

Choose what the structure physically is, not whichever overlay looks most
attractive. The recommendation is a warning aid, not automatic model authority.
If the app warns that another recipe is more compatible, review the image and
measurement objective before continuing.

Set credible minimum/maximum physical sizes. They are acceptance bounds, not a
request to force every contrast variation into an object. Set polarity and line
orientation explicitly when Auto is uncertain.

## 4. Guide and restrict detection

- **Draw one reference feature:** tightly enclose one complete, isolated and
  representative feature. The reference supplies scale/shape guidance; it does
  not impose identical dimensions on all objects.
- **Draw rectangular analysis region:** analyze a selected original-image area.
- **Draw freehand mask:** include or exclude an irregular region, labels or
  artifacts.
- **Reject partial border features:** prevents clipped objects from entering the
  released population.

## 5. Run and review

Press **Run model-based analysis**. Inspect the overlay before the table:

- the contour should follow the intended physical boundary;
- IDs should correspond to table rows;
- scale overlay and representative pitch/height gauges must be plausible;
- orange contact-split particles are REVIEW-only visible segments;
- cyan layer interfaces must be reviewed and unnecessary lines removed;
- rejected border features and masked annotations must not remain.

`RELEASE CANDIDATE` means software gates passed; it is not an accredited result.
`REVIEW` means operator disposition is required. Do not silently delete poor
objects only to improve CV—record the exclusion rule.

## 6. Interpret statistics

The statistical report always gives N. For N≥2 it reports mean, standard
deviation, 3σ, standard error, approximate 95% CI of the mean, median, min, max,
range and CV. Standard error/CI describe precision of the population mean under
the sample model; they are not a complete measurement uncertainty.

Class-aware results are used only when the population model has adequate sample
size and separation. Class A/B/C are ordered by primary size. Confirm assignments
before using class trends. Combined contact/isolated particle statistics remain
explicitly REVIEW.

## 7. Plot and inspect individuals

Select any available metric in **Distribution & control plots**. Hover a point
to view object/structure ID, value, class and centroid where available. Use the
histogram to identify multimodality and the individual plot to identify spatial
or acquisition-order trends. Investigate outliers physically; do not remove them
solely because they cross ±3σ.

## 8. Special workflows

### Cross-section population

Draw ROI 1 and run, then ROI 2 and run, continuing for each structure. Stored
ROIs remain overlaid and provide combined statistics. Height is from detected
feature top to the first sustained local substrate interface. Verify the green
height gauge endpoints.

### Layer stacks

Draw a yellow ROI through the stack. Interfaces I1… are ordered top-to-bottom;
layer L1 is I1–I2. Remove false candidates by interface ID or click-to-remove.
For a known repeat count, enter **Periods spanned by layer ROI**; otherwise Auto
requires sufficient repeated signal.

### Line profile

Draw through representative particle centers. The app reports calibrated
FWHM-like lateral chords, edge gaps and center spacing plus the raw intensity
profile. A chord approximates diameter only for a central transect. Intensity is
not topographic height.

### Process/dose study

Analyze each image with the same reviewed recipe and add it to the study with a
trusted dose/factor, condition, sample and replicate. Compare metrics common to
at least two images. Each point is one image result; error bars are within-image
feature standard deviation. Use replicates to assess image-to-image variation.

## 9. Export and report

CSV preserves numerical results. PNG exports the active plot. The professional
PDF can include the calibrated overlay, feature table, statistics, plot and QA
notes. Record the app version, recipe, calibration source, image identity,
operator decisions and exclusions in publications or regulated records.

## 10. Citation

Use the repository citation generated from `CITATION.cff` and cite the exact
software version. A methods statement should also name the recipe, calibration
source, edge/contour model, image sampling, population N and exclusion criteria.
