# SCL-RTK / RTKLIB-QR research branch

**Structured Covariance Learning for RTK**: C-native RTK with causal code-quality features, structured Q/R multipliers and integer-candidate-conditioned reference-position supervision. Original LAMBDA, acceptance and hold algorithms remain; no pseudorange bias or coordinate-correction branch.

Start with [`research/learned_qr/docs/SCL_RTK.md`](research/learned_qr/docs/SCL_RTK.md). General raw-data/CSV interfaces: [`TRAINING.md`](research/learned_qr/docs/TRAINING.md). Source and tests live in `research/learned_qr`; native hooks are already integrated under top-level `src`.

V1/V2 files remain readable. V3 fixes phase weights and uses causal observation consistency rather than absolute-C/N0 gating. Training supervises actual continuous FLOAT **and DGPS-labelled filter states**, not independent SINGLE/SPP outputs. Conditional gradients do not differentiate integer search, local H, feature construction or hold decisions.

A completed software test is not evidence that a trained model improves positioning. Validation rejection leaves an explicitly untrained identity fallback. Never deploy synthetic or rejected weights as a validated field model. Previous negative experiment reports are retained. This branch does not modify upstream `main`.
