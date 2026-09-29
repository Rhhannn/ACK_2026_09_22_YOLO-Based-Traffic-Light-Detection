# Package verification

Verification performed on 2026-09-29 before packaging:

- all Python files in the six code folders and `tests/` passed an AST syntax check;
- all five regression tests in `tests/test_core_metrics.py` passed;
- script 48, with 100,000 two-sided sequence-cluster sign-flip permutations
  and Holm correction, regenerated all four included CSV/JSON files byte for
  byte in two independent runs;
- the three staged model weights matched the sizes and SHA-256 checksums in
  `weights/README.md`;
- no author-machine absolute path or credential-like value was found in the
  repository copy.

The registered DTLD data are deliberately absent, so the image conversion,
training, and inference stages were not rerun as part of this packaging check.
Their scripts and exact split/configuration records are included for reviewers
who obtain DTLD from the official provider.
