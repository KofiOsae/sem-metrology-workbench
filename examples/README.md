# Tutorial samples

Run `python generate_validation_samples.py` to create four deterministic,
redistributable SEM-like images and `generated/ground_truth.json`.

Run `python validate_generated_samples.py` from the project root to check
calibration, circle populations, grating CD, irregular-particle count and the
cross-section height datum against disclosed tolerances.

These are synthetic training/regression fixtures—not literature images, physical
SEM acquisitions, or certified dimensional standards. Public external datasets
for robustness trials are cited in `docs/VALIDATION.md` and are not redistributed
here because users must preserve their source attribution and license.
