# Troubleshooting

## Scale or dimensions are wrong

- Verify the magenta endpoint dots, printed value and nm/µm unit.
- Use **Redraw scale-bar endpoints** when automatic detection includes label
  text, a footer baseline or extra ruler ticks.
- Prefer trusted companion pixel-size metadata and calibrated reference
  materials.
- Rerun analysis after any calibration change; do not reuse prior results.

## Too few objects are detected

- Confirm recipe and bright/dark polarity.
- Draw a complete isolated reference feature.
- Expand credible size limits only if true features lie outside them.
- Check exclusion mask, ROI and border-rejection behavior.
- Low contrast, blur, clipping and heavy overlap may be irrecoverable.

## Too many or false objects are detected

- Tighten the reference and credible size range.
- Exclude annotations, footer, charging artifacts and debris.
- Use the physical recipe matching the structure.
- Review overlay boundaries, not only the object count.

## Touching particles look undersized

Watershed boundaries estimate visible segments and cannot reconstruct hidden
overlap. Use isolated statistics, improve specimen dispersion/acquisition, or
manually restrict the analysis to interpretable particles.

## Cross-section height is implausible

The ROI must include a sustained local substrate interface and the complete top.
Avoid cap-layer texture and neighboring features. Verify the green datum gauge.
Measure multiple ROIs rather than enlarging one ROI over unrelated structures.

## Layer interface editing does not respond

Run the layer analysis first. Select an interface ID and press **Remove selected
interface** (most reliable), or activate click-to-remove and click inside the
yellow ROI near the cyan line. At least two outer interfaces are retained.

## Plots are empty

Run a model with at least one valid feature, select a numeric metric, and ensure
the lower panel is expanded. N=1 has no dispersion/control limits.

## Windows will not start the executable

- Download the x64 release asset again and verify SHA-256.
- Extract/run from a writable local folder rather than a network archive.
- Antivirus may quarantine unsigned PyInstaller applications; verify the hash
  and obtain the release only from the official repository.
- Report Windows version, exact error and release hash—never confidential data.

## Reporting a reproducible issue

Include app version, recipe/settings, calibration source, expected result,
overlay screenshot, and a de-identified shareable image or synthetic reproducer.
