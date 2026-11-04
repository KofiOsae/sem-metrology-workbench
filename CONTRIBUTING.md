# Contributing

Use an issue before changing a measurement definition. A metrology change must
state the physical quantity, datum, edge model, units, applicability, failure
modes and a synthetic or reference-data test with expected values.

Pull requests should include:

1. a focused description and representative overlay;
2. tests and their expected tolerance;
3. updated Help/user documentation;
4. no private SEM images or instrument metadata without permission;
5. confirmation that `python -m unittest discover -s tests -v` passes.

Security issues should follow `SECURITY.md` rather than a public issue.
