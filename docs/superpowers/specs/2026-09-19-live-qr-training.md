# Live native RTK Q/R training

User-approved scope: finish Q/R learning in research/learned-qr, preserve original
integer fixing, no bias head, deliver tools for the user's own data. Do not touch main.

The native solver remains authoritative for RINEX decoding, broadcast ephemerides,
current-state residual/Jacobian evaluation, satellite selection, state resets and AR.
A per-session callback tape mirrors continuous FLOAT state/covariance operations in
PyTorch and supplies neural scale factors in the forward C recursion. Raw measurements
are reprocessed at each training pass; archived residuals are never replayed as truth.

Training uses conditional first-order EKF derivatives: discrete selection and reset
choices, local H, ENU rotations and feature extraction are stop-gradient. The state,
covariance, ambiguity common-offset transform, Q/R scale factors and Kalman updates
remain connected across a truncated sequence. This is an explicit engineering
approximation, not the paper's exact architecture or an exact derivative through all
native decisions. Reference is used only by Python loss/evaluation, never C inference.

Supported initial training configuration: forward kinematic, dynamics on, one filter
iteration, non-IFLC observations, no estimated atmosphere or GLONASS autocal states,
broadcast ephemeris, no base residual interpolation or external calibration files.
Other configurations fail with actionable errors instead of silently changing models.
AR is disabled only during training and validation of FLOAT; deployment preserves it.

All deployment remains C99. Python is needed for training and optional evaluation.
Model load defaults off. Test/untrained weights require explicit opt-in. Training
session errors abort; there is no silent gradient/solver failure. Data stays local.
