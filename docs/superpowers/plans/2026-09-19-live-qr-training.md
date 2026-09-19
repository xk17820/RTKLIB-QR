# Live RTK Q/R Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Local RINEX + reference -> train Q/R -> export -> native RTK replay.
**Architecture:** Per-session native callbacks with a differentiable FLOAT tape;
keep native residual recalculation and state management, explicitly condition gradients
on the current linearization and discrete decisions. No frozen baseline residual replay.
**Tech Stack:** C99, ctypes, PyTorch float64, NumPy, pytest, CMake.
**Spec:** docs/superpowers/specs/2026-09-19-live-qr-training.md

## Global Constraints
- Preserve resamb_LAMBDA, manage_amb_LAMBDA, holdamb, ddcov and lambda.c byte-for-byte.
- No bias head, no reference coordinate as an input feature, no main branch changes.
- Forward short-baseline training profile only; unsupported settings raise.
- Use the current native state at every measurement call. Conditional EKF gradients
  and truncated BPTT must be named in documentation, never claimed exact full autodiff.

## Task 1: Integrate native inference
- [ ] RED: pytest tests/test_core_integration.py -> root adapter absent.
- [ ] Apply existing guarded transformations and copy headers to src/.
- [ ] GREEN: component suite, native build, protected function digest checks.

## Task 2: Native training bridge and callback tape
Files: research/learned_qr/src/qr_bridge.h, qr_bridge.c, qr_trace.h;
src/rtkpos.c, learned_qr_adapter.h; qrlearn/native.py, live.py.
Interfaces: qr_open/qr_step/qr_restart/qr_close; callback events carry borrowed
column-major arrays with explicit dimensions. Copy any needed data before returning.
- [ ] RED: integration test must fail for missing shared bridge entry points.
- [ ] Implement RINEX sessions, option guards, matched epochs and resets.
- [ ] Mirror scalar resets, position transition, Q additions, ambiguity diagonal noise,
  zero/reset/common-offset operations, SD factors and filter updates.
- [ ] GREEN: live forward parity, reference gradients for Q and R, bad config handling,
  callback exceptions, repeatability and session isolation.

## Task 3: Data, training and evaluation CLI
Files: qrlearn/reference.py, cli.py; configs/short-baseline.conf; example manifest.
- [ ] RED: tests for strict GPST/ECEF schema, interpolation gaps, no extrapolation,
  duplicate/nonfinite rows, invalid quality masks, no reference leakage.
- [ ] Add sequence passes, alternating optimizer steps, clipping, validation selection,
  safe tensor checkpoints, atomic native model export, position and FIX evaluation.
- [ ] GREEN: CLI raw-RINEX smoke with explicitly artificial test references, save/reload
  native model, reject empty supervision, train/validation route disjointness.

## Task 4: Verification and publish
- [ ] Full component suite; build integrated rnx2rtkp and bridge.
- [ ] Off/identity native regression on bundled RINEX, protected algorithm hash equality.
- [ ] Record actual scope and commands, no accuracy claims from artificial labels.
- [ ] Commit changed files to research/learned-qr; verify remote SHA and CI.
