# SEM Metrology Workbench

SEM Metrology Workbench is an operator-guided Windows desktop application for
traceable dimensional analysis of scanning electron microscope images. Version
1.0.0 is a standalone native Qt application; recipients do not need Python.

> **Measurement status:** this is research software, not a certified metrology
> instrument. Calibration, model compatibility, overlays and QA warnings must
> be reviewed by a competent operator. SEM grayscale is never treated as
> calibrated topographic height.

## Download and start

1. Download `SEM-Metrology-Workbench-v1.0.0-Windows-x64.exe` from the GitHub
   Releases page.
2. Verify the published SHA-256 checksum.
3. Run the executable. Windows SmartScreen may show an unrecognized-publisher
   warning until the binary is code-signed.
4. Open a raw SEM image, verify calibration, choose the physical recipe, review
   the overlay, and only then export results.

Complete instructions are in [User Guide](docs/USER_GUIDE.md),
[Tutorials](docs/TUTORIALS.md), and [Troubleshooting](docs/TROUBLESHOOTING.md).

## Released capabilities

- automatic scale-bar proposal, companion metadata calibration and manual
  scale-bar endpoint correction;
- circles, rings and holes with diameter, pitch, classes and edge roughness;
- rectangular/elongated periodic arrays with length, width, period and LER;
- irregular particles with isolated/contact-aware Feret and shape statistics;
- line-space gratings with CD, pitch, duty cycle, LER and LWR;
- single contours and curved/tapered waveguides;
- isolated cross-section features and reviewed layer stacks;
- calibrated line profiles reporting lateral chords, gaps and spacing;
- SEM intensity texture descriptors (Ra/Sa signal, explicitly not height);
- class-aware statistics, distributions, interactive point tooltips, QA and
  professional CSV/PDF/PNG export;
- multi-image process/dose studies.

See [Capability Matrix](docs/CAPABILITY_MATRIX.md) for definitions and limits.

## Validation

Run the deterministic unit and synthetic-metrology suite:

```powershell
python -m unittest discover -s tests -v
```

Generate the openly redistributable synthetic trial images:

```powershell
python examples/generate_validation_samples.py
```

The validation approach and public external datasets are listed in
[Validation](docs/VALIDATION.md). Passing tests verifies software behavior on
the disclosed cases; it does not establish measurement uncertainty for a
specific SEM, specimen, acquisition condition or laboratory.

## Privacy and usage statistics

The desktop application sends no telemetry and does not upload images or
results. Adoption is measured only from aggregate GitHub traffic and release
asset download counts. See [Privacy](PRIVACY.md).

## Citation

Use GitHub's **Cite this repository** button, which reads `CITATION.cff`. Add
the final public repository URL when the destination is known. After the first
public release is archived in Zenodo, add its DOI. Always include the version
used in the methods section.

## License and copyright

Copyright © 2026 KofiOsae. Source code is licensed under BSD-3-Clause. Documentation is licensed under
CC BY 4.0. Third-party packages retain their own licenses; see
[Third-party notices](THIRD_PARTY_NOTICES.md).
