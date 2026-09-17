# Feature schema V1

This is a deliberately reduced starter feature set, not the manuscript's complete
Table I or Fig. 1 architecture. Neither uncertainty targets nor reference coordinates
enter the inference features. All values must be finite.

## Temporal layout

Window = 10. Oldest first. Pad missing leading steps with zeros. Append a validity
scalar to every step: 0 for padding, 1 for a real step. Thus Q input = 10x11,
R input = 10x13. One tanh hidden layer with 16 units, followed by a tanh-bounded
log-variance scale. The same dense row-major weights are used by Python and C.

Each R history is keyed by (RTK instance,satellite,frequency slot,code/phase).
Rover AND base observation codes form the signal-continuity signature. Either
receiver's LLI slip or RTKLIB's aggregate slip flag resets that signal at a new epoch.
Repeated invocations at the same epoch and same signature reuse cached scaling;
postfit/iteration calls cannot repeatedly ingest the same epoch or alter the scale.
A missing period longer than the model max_gap (default 30 seconds) resets history.
Backward time resets the generic history object; the RTK adapter bypasses learning
in a backward pass. A position reinitialization clears the whole filter's QR history.

## Q input (after F propagation, before adding acceleration driving covariance)

0. log1p(dt_seconds), dt > 0
1–3. predicted velocity in ENU / 30 m/s
4–6. predicted acceleration in ENU / 3 m/s^2
7. log1p(sqrt(mean(diagonal(P_position))))
8. currently available `rtk.sol.ns` / 40
9. currently available `rtk.sol.stat` / 6

Q outputs horizontal and vertical VARIANCE scale. Native RTKLIB adds process
noise only to its acceleration block in this function. We preserve that placement
and the native ENU->ECEF rotation. We do NOT scale P itself.

## R input at first use of a signal in an epoch

0. (rover C/N0 - 40) / 10
1. (base C/N0 - 40) / 10
2. sin(rover elevation in radians)
3. 0 = phase, 1 = code
4. zero-based frequency slot / max(compiled NFREQ - 1, 1)
5. constellation index / 6: GPS,GLO,GAL,BDS,QZSS,SBAS,NavIC = 0..6
6. log1p(abs(baseline metres) / 1000)
7. sign(age)*log1p(abs(base-rover observation age in seconds))
8. sign(reported_std)*log1p(abs(reported_std)); same obs.Pstd or obs.Lstd value
   that the receiver parser exposes, not a newly inferred physical standard deviation
9. half-cycle flag on either receiver, 0/1
10. change in normalized rover C/N0 relative to the last continuous epoch; 0 after reset
11. log1p(continuous tracked epoch count); count = 1 after reset

R outputs a single multiplier for that station-single-difference variance in m^2.
The existing varerr() computes the baseline SD variance, including its existing
receiver/baseline/clock/SNR factors. We do NOT add another rover+base factor of 2.
The existing half-cycle additive variance remains outside this multiplier.
Reference selection and DD covariance formation see the same cached signal scale.

## Explicit omissions and limitations

No innovation/postfit residual feature is used in V1. In RTKLIB these residuals can
be relative to a changing DD reference satellite, and ddres() overwrites the stored
residuals during iterations. Adding such features without correct reference and
update semantics would make training/deployment inconsistent. A V2 design may
add reference-consistent residuals and pseudorange–Doppler discrepancy, but must
change the schema/version and retrain. V1 also does not use attention, a full Q
Cholesky factor, ephemeris features, or any ambiguity-domain neural head.

The log contains GPST seconds since GPS epoch as returned by time2gpst, not UTC
wall time. It is only a feature log; it cannot reconstruct full filtering training.
