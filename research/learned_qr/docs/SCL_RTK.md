# SCL-RTK — Structured Covariance Learning for RTK

This research implementation learns motion-process and code-observation covariance multipliers while retaining the native RTKLIB ambiguity search, acceptance and hold routines. It does not predict coordinates or introduce a pseudorange bias head. Software correctness and learned positioning performance are separate claims; a model rejected by validation is not promoted as a successful model.

## Mechanism

- Native C forward and deployment; 10-epoch causal feature windows and 16-unit temporal MLPs.
- Q scales the original horizontal/vertical acceleration driving covariance, not every state or ambiguity variance.
- V3 R uses clock-centred standardized code innovations and dual-frequency code–carrier multipath deviation. C/N0 remains an input, not the quality gate. The MP feature has a causal, bounded-rate anchor and resets on **either** frequency's discontinuity.
- Code inflation is `1 + g * max(s_theta - 1, 0)`. Phase weights remain exactly at their original values. V3 rejects a header claiming a different phase cap.
- SD variances are transformed by the native DD covariance constructor, retaining reference-satellite correlations and existing half-cycle additions.
- The read-only candidate observer copies native flags before calling the original DD selector/LAMBDA. It provides an integer vector before ratio acceptance for the conditional loss `p_check = p - P_pa solve(P_aa, a - a_check)`.
- A separate read-only observer exports **actually accepted** DD integers for simulation audit. Position-threshold exceedance is not used as a synonym for an incorrectly fixed integer.

## Training and derivative scope

The default experiment uses native continuous FLOAT/DGPS forward states. RTKLIB may label an accepted continuous update as DGPS when fewer than four phase satellites remain; these states are still `rtk->x` and **are supervised**. Independent SINGLE/SPP outputs are not attached to a fictitious filter gradient. Reference positions are never passed as native observations, features, or initial states.

The compact tape retains active-state means/covariances, SD/DD covariance dependence, Kalman gain and Joseph updates. It stops gradients through local H, feature extraction, rotations, discrete activity/selection/reset decisions and integer search. The conditional candidate position is differentiable **given a fixed integer vector**. This is conditional first-order differentiation, not exact differentiation of the entire C program or fix-and-hold decisions. Main simulations use native continuous AR; unchanged hold code is not evidence of validated hold-path derivatives.

Optional deterministic rotating-window TBPTT (`--gradient-stride 4`) differentiates one in four 64-epoch blocks, while C propagates **every** raw observation. Reloading model weights does not reset the native state/history. Raw-forward, supervised and gradient-bearing epoch counts are recorded separately. `--gradient-stride 1` differentiates every block. A six-round/stride-four experiment does not imply each branch has differentiated every epoch.

The objective combines ENU Huber position error, conditional-candidate position error, innovation NLL, log-multiplier deviation and within-window temporal regularization. Q/R stages alternate. Invalid references mask only the objective. Validation uses complete native RTK; no qualified checkpoint means `best.qr` remains an explicitly untrained identity model.

## Run with local raw data

Rebuild every dependent module: the branch adds native context and observer interfaces.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE='-O2 -UNDEBUG'
cmake --build build -j4
ctest --test-dir build --output-on-failure
python -m pip install -r research/learned_qr/requirements.txt
(cd research/learned_qr && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q)
python research/learned_qr/run_qr.py train \
  --manifest /data/dataset.json --config /data/short-baseline.conf \
  --output outputs/scl01 --r-policy scl --compact \
  --conditional-weight 1 --smooth-weight 0.0001 \
  --epochs 6 --sequence-length 64 --gradient-stride 4
```

Manifest/reference formats and legacy replay commands remain in `TRAINING.md`. Use GPS dual-frequency, short-baseline forward dynamics for the verified SCL study; multi-constellation transfer, long baselines, antenna/bias changes and real hardware are not certified by these simulations. Study weights retain `synthetic` provenance and require explicit test opt-in.

## Native APIs and checks

`qr_set_learning_options` enables V3 features/read-only candidate or accepted-integer observers. `qr_reload_model` switches between native model inference and a callback without state reset. Callback arrays are borrowed only during that call. The caller may change inference factors only, never source state/covariance arrays.

Regression tests cover native/Torch conditional position equality, two-step Q/R finite differences with a fixed integer, export parity, identity reload, active-state tape equality, L2-only MP reset, rejected malformed models, and the FLOAT/DGPS supervision distinction. These tests establish their stated contracts, not field accuracy or universal robustness.
