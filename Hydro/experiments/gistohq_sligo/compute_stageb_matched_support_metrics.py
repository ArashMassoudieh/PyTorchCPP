#!/usr/bin/env python3
"""Compute matched-support control-vs-multiscale metrics for Stage B
(improvement_loop_20261023), with the same fail-closed row validation
discipline as select_multiscale_history_length.py's read_validation_rows,
generalized to an arbitrary split (here: 'test', the primary comparison
per CLAUDE_SUPERVISOR_TASKS.md's 2026-11-16 action checklist -- validation
is selection-conditioned, see checkpoint 63/report 94).

For each (fold, seed) pair: reads both arms' own predictions.csv, restricted
to rows actually labeled with the requested split AND independently
cross-checked against that fold's own chronological interval (the label
alone is never trusted). Rejects non-finite/non-integer/sealed/duplicate/
ambiguous-split timestamps before any value is read, exactly as
read_validation_rows does for 'validation'. Confirms the two arms' eligible
x-sets are IDENTICAL (not merely equal in count) and that observed values
agree at every shared timestamp, before computing any metric. Records
sha256 of every input artifact (both arms' predictions.csv/experiment_
config.json/checkpoint, plus the staged input config and sidecar each
job's boundary declaration came from) and a sha256 of the sorted matched-x
list itself, so a reviewer can verify this exact invocation independently
reproduces the same support and metrics it claims.

Writes one explicit JSON report; prints nothing but a path on success.
Reads only already-exported artifacts; starts no training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

RESERVED_PERIOD_X_CUTOFF = 77016
ALLOWED_SPLITS = ("test", "validation")
STAGE_B_OUTPUT = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "improvement_loop_20261023/multiscale_comparison_stage_b_output"
)
FROZEN_MANIFEST_HASHES = Path(
    "/mnt/3rd900/Projects/PyTorchCPP/build-hydrobatch/multiscale_stage_b_manifests_staged/"
    "STAGED_MANIFEST_HASHES.txt"
)
NATIVE_DISCHARGE_SUPPORT = Path(
    "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/"
    "native_discharge_support_v2.json"
)
CATCHMENT_AREA_KM2 = 19.0893  # dataset_manifest.json's catchment_area_m2=19089300.0; actually
# checked (not merely claimed) against each job's own dataset_manifest.json by
# _verify_catchment_area(), called from get_native_matched_rows() for both arms (checkpoint67 --
# the prior comment claimed this check happened in main() when no such check existed anywhere).
RESOLVED_D = {"2018": 30, "2020": 30, "2023": 60}
SEEDS = ("42", "123", "27182")
STUDY_START_UTC = "2015-01-01T00:00:00Z"  # x=0; confirmed identical across all 18 Stage B job
# dirs' own dataset_manifest.json (checkpoint67) -- x is an integer hour-index from this epoch,
# confirmed by converting each fold's own test_end_x into a date and finding it lands exactly on
# that fold's name year (e.g. 2018 fold's test_end_x=35063 -> 2018-12-31T23:00Z) and on the 2023
# fold's manifest-declared study_end (2023-10-14T23:00Z, one hour before the reserved-period seal).


def _parse_frozen_manifest_hashes(path: Path) -> Dict[str, str]:
    """Parse a 'ROOT: <dir>' + '<sha256>  <path>' manifest-hash file into
    {resolved_absolute_path: expected_sha256}. Matches the format written
    by this program's own launcher scripts (run_multiscale_selection_
    batch.sh): a bare filename resolves relative to ROOT; an absolute path
    is used as-is."""
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines or not lines[0].startswith("ROOT:"):
        raise MatchedSupportError(f"{path} must start with a 'ROOT: <dir>' line")
    root = Path(lines[0][len("ROOT:"):].strip()).resolve()
    expected: Dict[str, str] = {}
    for line in lines[1:]:
        digest, _, raw_path = line.partition("  ")
        candidate = Path(raw_path) if Path(raw_path).is_absolute() else root / raw_path
        expected[str(candidate.resolve())] = digest
    return expected


def _load_native_discharge_support() -> set:
    """Validate each coordinate BEFORE truncating it with int() -- int(1500.7)
    silently discards the .7 instead of rejecting a value that should never
    occur in an hour-index field (checkpoint67: the prior version truncated
    unconditionally, which would have silently accepted a corrupted
    non-integer coordinate instead of refusing it)."""
    data = json.loads(NATIVE_DISCHARGE_SUPPORT.read_text())
    result = set()
    for raw in data["strict_observed_hour_x"]:
        x = float(raw)
        if not math.isfinite(x) or x != int(x) or x < 0:
            raise MatchedSupportError(
                f"{NATIVE_DISCHARGE_SUPPORT}: strict_observed_hour_x contains a non-finite, "
                f"non-integer, or negative value {raw!r} -- refusing to silently truncate it"
            )
        result.add(int(x))
    return result


class MatchedSupportError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_split_rows_fail_closed(
    predictions_csv: Path,
    approach: str,
    *,
    split: str,
    interval_start_x: float,
    interval_end_x: float,
) -> Dict[float, Tuple[float, float]]:
    """Same fail-closed discipline as select_multiscale_history_length.py's
    read_validation_rows, generalized to any split name and its own
    (interval_start_x, interval_end_x] chronological interval."""
    rows: Dict[float, Tuple[float, float]] = {}
    x_to_split: Dict[float, str] = {}
    with open(predictions_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            # Timestamp guards apply to EVERY row, regardless of which
            # approach it belongs to -- reject a non-finite/non-integer/
            # sealed x BEFORE the approach filter, not after. A corrupted
            # or reserved-period row for some *other* approach sharing this
            # same predictions.csv must not be silently skipped past these
            # checks just because it isn't the approach being scored.
            x = float(row["x"])
            if not math.isfinite(x):
                raise MatchedSupportError(f"Non-finite x={x} in {predictions_csv} (approach={row.get('approach')!r})")
            if x != int(x):
                raise MatchedSupportError(
                    f"Non-integer x={x} in {predictions_csv} (approach={row.get('approach')!r}) -- "
                    "hour-index timestamps must be whole numbers"
                )
            if x < 0:
                raise MatchedSupportError(
                    f"Negative x={x} in {predictions_csv} (approach={row.get('approach')!r}) -- "
                    "hour-index timestamps must be nonnegative"
                )
            if x >= RESERVED_PERIOD_X_CUTOFF:
                raise MatchedSupportError(
                    f"x={x} is at or past the reserved-period seal ({RESERVED_PERIOD_X_CUTOFF}) "
                    f"in {predictions_csv} (approach={row.get('approach')!r}) -- refusing to "
                    "process, regardless of approach or split"
                )
            if row["approach"] != approach:
                continue
            row_split = row["split"]
            if x in x_to_split and x_to_split[x] != row_split:
                raise MatchedSupportError(
                    f"x={x} for approach {approach!r} in {predictions_csv} is labeled both "
                    f"{x_to_split[x]!r} and {row_split!r} -- ambiguous split assignment"
                )
            x_to_split[x] = row_split
            if row_split != split:
                continue
            if not (interval_start_x < x <= interval_end_x):
                raise MatchedSupportError(
                    f"x={x} for approach {approach!r} in {predictions_csv} is labeled {split!r} "
                    f"but falls outside the fold's own declared interval "
                    f"({interval_start_x}, {interval_end_x}]"
                )
            if x in rows:
                raise MatchedSupportError(f"Duplicate {split} x={x} for approach {approach!r} in {predictions_csv}")
            observed_value = float(row["observed"])
            predicted_value = float(row["predicted"])
            if not math.isfinite(observed_value) or not math.isfinite(predicted_value):
                raise MatchedSupportError(
                    f"Non-finite value at x={x} for approach {approach!r} in {predictions_csv}"
                )
            rows[x] = (observed_value, predicted_value)
    if not rows:
        raise MatchedSupportError(f"No {split} rows found for approach {approach!r} in {predictions_csv}")
    return rows


def nse(observed, predicted) -> float:
    n = len(observed)
    mean_obs = sum(observed) / n
    denom = sum((o - mean_obs) ** 2 for o in observed)
    if denom <= 0.0:
        raise MatchedSupportError("NSE denominator is zero or negative")
    numer = sum((o - p) ** 2 for o, p in zip(observed, predicted))
    return 1.0 - numer / denom


def kge(observed, predicted) -> float:
    n = len(observed)
    mean_o = sum(observed) / n
    mean_p = sum(predicted) / n
    std_o = math.sqrt(sum((o - mean_o) ** 2 for o in observed) / n)
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in predicted) / n)
    if std_o == 0 or std_p == 0 or mean_o == 0:
        raise MatchedSupportError("KGE undefined: zero observed/predicted variance or zero observed mean")
    cov = sum((o - mean_o) * (p - mean_p) for o, p in zip(observed, predicted)) / n
    r = cov / (std_o * std_p)
    alpha = std_p / std_o
    beta = mean_p / mean_o
    return 1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(observed, predicted) -> float:
    n = len(observed)
    return math.sqrt(sum((o - p) ** 2 for o, p in zip(observed, predicted)) / n)


def mae(observed, predicted) -> float:
    n = len(observed)
    return sum(abs(o - p) for o, p in zip(observed, predicted)) / n


def signed_bias_percent(observed, predicted) -> float:
    sum_o = sum(observed)
    if sum_o == 0:
        raise MatchedSupportError("PBIAS undefined: sum of observed values is zero")
    return 100.0 * sum(p - o for o, p in zip(observed, predicted)) / sum_o


def absolute_bias_percent(observed, predicted) -> float:
    sum_o = sum(observed)
    if sum_o == 0:
        raise MatchedSupportError("Absolute-bias-% undefined: sum of observed values is zero")
    return 100.0 * abs(sum(p - o for o, p in zip(observed, predicted))) / sum_o


def all_metrics(observed, predicted) -> dict:
    """observed/predicted are in mm/h (the frozen internal training/export
    unit -- confirmed, not assumed, by the harmonizer target construction).
    RMSE/MAE are also reported in m^3/s via Q[m3/s] = depth[mm/h] *
    area[km2] / 3.6, the same conversion and area (19.0893 km^2, confirmed
    against each job's own dataset_manifest.json by the caller) used
    throughout this program -- both unit representations are kept, the raw
    mm/h value is never discarded or replaced. PBIAS is NOT unit-dependent
    (a ratio of sums), so no m^3/s variant is needed for it. Signed
    PBIAS = 100*sum(predicted-observed)/sum(observed); absolute PBIAS is
    exactly abs(signed PBIAS) for a positive observed-volume sum -- not MAE,
    which is a different, pointwise-absolute-error quantity."""
    rmse_mm_h = rmse(observed, predicted)
    mae_mm_h = mae(observed, predicted)
    unit_scale = CATCHMENT_AREA_KM2 / 3.6
    return {
        "nse": nse(observed, predicted),
        "kge": kge(observed, predicted),
        "rmse_mm_h": rmse_mm_h,
        "rmse_m3_s": rmse_mm_h * unit_scale,
        "mae_mm_h": mae_mm_h,
        "mae_m3_s": mae_mm_h * unit_scale,
        "pbias_signed_percent": signed_bias_percent(observed, predicted),
        "pbias_absolute_percent": absolute_bias_percent(observed, predicted),
    }


def _verify_catchment_area(job_dir: Path, *, label: str) -> None:
    """Actually check (not merely claim in a comment) that this job's own
    dataset_manifest.json declares the same catchment area as
    CATCHMENT_AREA_KM2, the constant all_metrics() uses for the mm/h->m^3/s
    conversion. Fixes checkpoint67: the module-level comment on
    CATCHMENT_AREA_KM2 said this check happened in main(), but no such check
    existed anywhere in the file -- a false claim, independently caught."""
    manifest_path = job_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    area_m2 = manifest.get("catchment_area_m2")
    if area_m2 is None or not isinstance(area_m2, (int, float)) or not math.isfinite(area_m2) or area_m2 <= 0:
        raise MatchedSupportError(f"{label}: {manifest_path} has a missing or invalid catchment_area_m2")
    expected_m2 = CATCHMENT_AREA_KM2 * 1_000_000.0
    if abs(area_m2 - expected_m2) > 1.0:
        raise MatchedSupportError(
            f"{label}: {manifest_path}'s catchment_area_m2={area_m2} does not match the "
            f"CATCHMENT_AREA_KM2 constant used for the m^3/s conversion ({expected_m2} m^2) -- "
            "the reported m^3/s values would be silently wrong for this job"
        )


def _verify_sidecar(
    job_dir: Path,
    sidecar: dict,
    *,
    label: str,
    expected_d: int,
    expected_matched_control: bool,
    expected_seed: int,
    frozen_manifest_hashes: Dict[str, str],
    expected_year: Optional[str] = None,
) -> dict:
    """Re-verify a boundary_provenance.json sidecar against the files it
    actually describes AND against independent anchors, right now -- a
    sidecar's own recorded hashes/fields are a claim about what was true
    when it was written, not evidence about current reality. Fixes two
    gaps found on independent review (2026-11-16): (1) the prior version
    only hash-checked the staged input file as an opaque blob without ever
    parsing its content to confirm declared_from_staged_input's VALUES
    actually match what's in that file -- a sidecar could declare
    fabricated bounds while still passing a hash check against an
    unrelated, genuinely-unmodified file; (2) the staged file's hash was
    only compared against the sidecar's own self-reported claim, never
    against the independent FROZEN_MANIFEST_HASHES record, so a sidecar
    and its referenced file could agree with each other while both having
    silently drifted from the frozen manifest. Also fixes a third gap found
    on 2026-11-17 review: the identity check previously only confirmed the
    exported and staged configs' random_seed agreed WITH EACH OTHER, not
    that either matched the caller's actual expectation -- a job-directory
    pair whose exported+staged files were swapped together (e.g. with
    another fold/seed's files) could still satisfy mutual equality while
    being entirely the wrong pair. Now requires both files' random_seed to
    equal `expected_seed` explicitly, and (when `expected_year` is given)
    both files' experiment_id to contain that year string -- catching a
    whole-directory fold swap that `expected_d` alone cannot, since two
    folds can share the same D (2018 and 2020 both resolved D=30).

    Returns the freshly-parsed {train_end_x, validation_end_x, test_end_x}
    (not the sidecar's own copy) so callers use what was actually just
    verified, not merely a value that happened to match it."""
    exported_path = Path(sidecar["exported_config_path"])
    staged_path = Path(sidecar["staged_input_config_path"])
    if exported_path.parent != job_dir:
        raise MatchedSupportError(
            f"{label}: sidecar's exported_config_path {exported_path} does not point inside "
            f"this job's own directory {job_dir}"
        )
    actual_exported_hash = _sha256_file(exported_path)
    if actual_exported_hash != sidecar["exported_config_sha256"]:
        raise MatchedSupportError(
            f"{label}: sidecar's declared exported_config_sha256 "
            f"({sidecar['exported_config_sha256']}) does not match the actual current file "
            f"({actual_exported_hash}) -- the exported config changed since the sidecar was "
            "written, or the sidecar itself is stale/wrong"
        )
    actual_staged_hash = _sha256_file(staged_path)
    if actual_staged_hash != sidecar["staged_input_config_sha256"]:
        raise MatchedSupportError(
            f"{label}: sidecar's declared staged_input_config_sha256 "
            f"({sidecar['staged_input_config_sha256']}) does not match the actual current "
            f"staged input file ({actual_staged_hash})"
        )

    # Independent anchor: the staged file's hash must also match the
    # frozen manifest, not merely the sidecar's own self-reported claim --
    # two mutually-consistent-but-wrong values would otherwise pass.
    frozen_key = str(staged_path.resolve())
    frozen_expected_hash = frozen_manifest_hashes.get(frozen_key)
    if frozen_expected_hash is None:
        raise MatchedSupportError(
            f"{label}: staged input {staged_path} is not present in the frozen manifest hash "
            "file at all -- cannot independently verify it against anything"
        )
    if frozen_expected_hash != actual_staged_hash:
        raise MatchedSupportError(
            f"{label}: staged input {staged_path}'s actual hash ({actual_staged_hash}) does not "
            f"match the independently frozen manifest's expected hash ({frozen_expected_hash})"
        )

    # Parse the staged file's REAL content -- do not trust the sidecar's
    # own declared_from_staged_input dict as a stand-in for what the file
    # actually contains.
    staged_content = json.loads(staged_path.read_text())
    actual_bounds = {
        "train_end_x": staged_content.get("train_end_x"),
        "validation_end_x": staged_content.get("validation_end_x"),
        "test_end_x": staged_content.get("test_end_x"),
    }
    if any(v is None for v in actual_bounds.values()):
        raise MatchedSupportError(f"{label}: staged input {staged_path} is missing one or more end_x fields")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v == int(v) for v in actual_bounds.values()):
        raise MatchedSupportError(f"{label}: staged input {staged_path}'s end_x fields must be finite integers")
    if actual_bounds["train_end_x"] < 0:
        raise MatchedSupportError(
            f"{label}: staged input {staged_path}'s train_end_x={actual_bounds['train_end_x']} is "
            "negative -- hour-index bounds must be nonnegative"
        )
    if not (actual_bounds["train_end_x"] < actual_bounds["validation_end_x"] < actual_bounds["test_end_x"] < RESERVED_PERIOD_X_CUTOFF):
        raise MatchedSupportError(
            f"{label}: staged input {staged_path}'s bounds are not chronological and before the "
            f"reserved-period seal: {actual_bounds}"
        )
    declared_bounds = sidecar["declared_from_staged_input"]
    if actual_bounds != declared_bounds:
        raise MatchedSupportError(
            f"{label}: sidecar's declared_from_staged_input ({declared_bounds}) does not match "
            f"the staged input file's actual parsed content ({actual_bounds}) -- the sidecar "
            "declaration is fabricated or stale, independent of whether its hash matched"
        )

    # D/seed/matched-control identity, freshly re-derived from both files,
    # not trusted from the sidecar's own "cross_checked_fields_agree" copy.
    exported_content = json.loads(exported_path.read_text())
    for field, expected in (
        ("multiscale_daily_history_days", expected_d),
        ("multiscale_matched_control", expected_matched_control),
    ):
        exported_val = exported_content.get(field, False if field == "multiscale_matched_control" else None)
        staged_val = staged_content.get(field, False if field == "multiscale_matched_control" else None)
        if exported_val != expected or staged_val != expected:
            raise MatchedSupportError(
                f"{label}: {field} identity mismatch -- expected {expected!r}, exported config "
                f"has {exported_val!r}, staged input has {staged_val!r}"
            )
    # Check against the CALLER'S expectation, not merely that the two files
    # agree with each other -- a pair swapped together (e.g. this job's
    # exported+staged files actually belonging to a different fold/seed)
    # would still satisfy mutual equality while being the wrong pair
    # entirely (2026-11-17 review).
    for source_name, content in (("exported config", exported_content), ("staged input", staged_content)):
        if content.get("random_seed") != expected_seed:
            raise MatchedSupportError(
                f"{label}: {source_name}'s random_seed ({content.get('random_seed')!r}) does not "
                f"match the expected seed {expected_seed!r} for this job directory -- mutual "
                "agreement between exported and staged files is not sufficient if both were "
                "swapped in together from a different job"
            )
    if expected_year is not None:
        for source_name, content in (("exported config", exported_content), ("staged input", staged_content)):
            experiment_id = content.get("experiment_id", "")
            if expected_year not in experiment_id:
                raise MatchedSupportError(
                    f"{label}: {source_name}'s experiment_id ({experiment_id!r}) does not contain "
                    f"the expected fold year {expected_year!r} -- possible whole-directory fold "
                    "swap (two folds can share the same D, so expected_d alone cannot catch this)"
                )

    return actual_bounds


def get_native_matched_rows(
    control_dir: Path,
    multiscale_dir: Path,
    *,
    split: str,
    expected_d: int,
    expected_seed: int,
    frozen_manifest_hashes: Dict[str, str],
    native_support: set,
    control_approach: str = "lstm_pinn",
    multiscale_approach: str = "lstm_multiscale",
    label: str = "",
    expected_year: Optional[str] = None,
) -> dict:
    """Single verified path to a matched, native-restricted (x, observed,
    control_predicted, multiscale_predicted) sample for one fold/seed pair --
    every consumer (compare_fold_seed_pair's metrics, and any later per-year/
    cross-seed/tail/event diagnostic) must go through this, not re-implement
    the verification independently, so a future diagnostic cannot silently
    diverge from what the published matched-support numbers are actually
    built on.

    Returns {"bounds", "control_sidecar", "multiscale_sidecar",
    "eligible_matched_x", "matched_x", "matched_x_sha256", "obs",
    "control_pred", "multiscale_pred"}; raises MatchedSupportError on any of
    the checks compare_fold_seed_pair has always enforced (sidecar/manifest/
    identity verification, exact eligible-set equality, native-support
    restriction)."""
    if split not in ALLOWED_SPLITS:
        raise MatchedSupportError(f"{label}: split {split!r} is not one of {ALLOWED_SPLITS!r}")

    _verify_catchment_area(control_dir, label=f"{label} control")
    _verify_catchment_area(multiscale_dir, label=f"{label} multiscale")

    control_sidecar = json.loads((control_dir / "boundary_provenance.json").read_text())
    multiscale_sidecar = json.loads((multiscale_dir / "boundary_provenance.json").read_text())
    control_bounds = _verify_sidecar(
        control_dir, control_sidecar, label=f"{label} control",
        expected_d=expected_d, expected_matched_control=True, expected_seed=expected_seed,
        frozen_manifest_hashes=frozen_manifest_hashes, expected_year=expected_year,
    )
    multiscale_bounds = _verify_sidecar(
        multiscale_dir, multiscale_sidecar, label=f"{label} multiscale",
        expected_d=expected_d, expected_matched_control=False, expected_seed=expected_seed,
        frozen_manifest_hashes=frozen_manifest_hashes, expected_year=expected_year,
    )
    if control_bounds != multiscale_bounds:
        raise MatchedSupportError(
            f"{label}: control and multiscale staged inputs' actual parsed bounds differ "
            f"({control_bounds} vs {multiscale_bounds}) -- matched jobs must share identical "
            "fold boundaries"
        )
    bounds = control_bounds

    if split == "test":
        interval_start_x, interval_end_x = bounds["validation_end_x"], bounds["test_end_x"]
    else:
        interval_start_x, interval_end_x = bounds["train_end_x"], bounds["validation_end_x"]

    control_rows = read_split_rows_fail_closed(
        control_dir / "predictions.csv", control_approach, split=split,
        interval_start_x=interval_start_x, interval_end_x=interval_end_x,
    )
    multiscale_rows = read_split_rows_fail_closed(
        multiscale_dir / "predictions.csv", multiscale_approach, split=split,
        interval_start_x=interval_start_x, interval_end_x=interval_end_x,
    )

    control_x = set(control_rows)
    multiscale_x = set(multiscale_rows)
    if control_x != multiscale_x:
        # Refuse, don't silently fall back to the intersection. The
        # matched-control design (report 71-72) is supposed to make these
        # sets identical by construction; a mismatch means that guarantee
        # broke somewhere, and scoring on whatever subset happens to
        # overlap would quietly launder a real support-mismatch bug into a
        # number that looks legitimate.
        raise MatchedSupportError(
            f"{label}/{split}: control and multiscale eligible-{split}-x sets are NOT identical "
            f"(control-only: {len(control_x - multiscale_x)}, "
            f"multiscale-only: {len(multiscale_x - control_x)}) -- refusing to score on the "
            "intersection; matched support must hold exactly, not approximately"
        )
    eligible_matched_x = sorted(control_x)
    for x in eligible_matched_x:
        if control_rows[x][0] != multiscale_rows[x][0]:
            raise MatchedSupportError(f"{label}: observed mismatch at x={x}")

    # Equal eligible-x sets are not by themselves evidence of NATIVE
    # observation support -- both arms could share the same interpolated/
    # non-native target hours and still agree exactly (2026-11-16 review).
    # Restrict actual scoring to the intersection with the frozen,
    # independently-established native_discharge_support_v2.json; disclose
    # both counts rather than silently treating eligible equality as
    # native truth.
    native_matched_x = sorted(x for x in eligible_matched_x if int(x) in native_support)
    if not native_matched_x:
        raise MatchedSupportError(
            f"{label}/{split}: none of the {len(eligible_matched_x)} eligible-matched x values "
            "are in the frozen native-discharge-observation support -- refusing to score on "
            "purely non-native/interpolated targets"
        )
    matched_x = native_matched_x

    obs = [control_rows[x][0] for x in matched_x]
    pred_c = [control_rows[x][1] for x in matched_x]
    pred_m = [multiscale_rows[x][1] for x in matched_x]

    matched_x_sha256 = hashlib.sha256(",".join(str(int(x)) for x in matched_x).encode()).hexdigest()

    return {
        "bounds": bounds,
        "control_sidecar": control_sidecar,
        "multiscale_sidecar": multiscale_sidecar,
        "eligible_matched_x": eligible_matched_x,
        "matched_x": matched_x,
        "matched_x_sha256": matched_x_sha256,
        "obs": obs,
        "control_pred": pred_c,
        "multiscale_pred": pred_m,
    }


def compare_fold_seed_pair(
    control_dir: Path,
    multiscale_dir: Path,
    *,
    split: str,
    expected_d: int,
    expected_seed: int,
    frozen_manifest_hashes: Dict[str, str],
    native_support: set,
    control_approach: str = "lstm_pinn",
    multiscale_approach: str = "lstm_multiscale",
    control_checkpoint: str = "lstm_pinn.pt",
    multiscale_checkpoint: str = "lstm_multiscale.pt",
    label: str = "",
    expected_year: Optional[str] = None,
) -> dict:
    """Compute matched-support metrics for one control/multiscale job pair.
    Extracted from main()'s loop body so it is directly unit-testable
    against synthetic job directories, not only against the real Stage B
    output. `frozen_manifest_hashes` and `native_support` are passed in
    (not loaded internally) so tests can supply synthetic versions instead
    of the real Stage B/native-discharge files."""
    r = get_native_matched_rows(
        control_dir, multiscale_dir, split=split, expected_d=expected_d, expected_seed=expected_seed,
        frozen_manifest_hashes=frozen_manifest_hashes, native_support=native_support,
        control_approach=control_approach, multiscale_approach=multiscale_approach, label=label,
        expected_year=expected_year,
    )
    control_sidecar, multiscale_sidecar = r["control_sidecar"], r["multiscale_sidecar"]
    obs, pred_c, pred_m = r["obs"], r["control_pred"], r["multiscale_pred"]

    return {
        "split": split,
        "n_eligible_matched": len(r["eligible_matched_x"]),
        "n_native_matched": len(r["matched_x"]),
        "n_matched": len(r["matched_x"]),  # scored support == native-matched, not merely eligible
        "support_matches_exactly": True,  # guaranteed: raised above otherwise
        "matched_x_sha256": r["matched_x_sha256"],
        "observed_values_agree_at_every_shared_x": True,
        "sidecars_independently_reverified_against_current_files": True,
        "sidecars_content_parsed_and_identity_checked": True,
        "control": all_metrics(obs, pred_c),
        "multiscale": all_metrics(obs, pred_m),
        "provenance": {
            "control_predictions_sha256": _sha256_file(control_dir / "predictions.csv"),
            "control_experiment_config_sha256": _sha256_file(control_dir / "experiment_config.json"),
            "control_checkpoint_sha256": _sha256_file(control_dir / "models" / control_checkpoint),
            "control_staged_input_path": control_sidecar["staged_input_config_path"],
            "control_staged_input_sha256": _sha256_file(Path(control_sidecar["staged_input_config_path"])),
            "multiscale_predictions_sha256": _sha256_file(multiscale_dir / "predictions.csv"),
            "multiscale_experiment_config_sha256": _sha256_file(multiscale_dir / "experiment_config.json"),
            "multiscale_checkpoint_sha256": _sha256_file(multiscale_dir / "models" / multiscale_checkpoint),
            "multiscale_staged_input_path": multiscale_sidecar["staged_input_config_path"],
            "multiscale_staged_input_sha256": _sha256_file(Path(multiscale_sidecar["staged_input_config_path"])),
            "control_sidecar_sha256": _sha256_file(control_dir / "boundary_provenance.json"),
            "multiscale_sidecar_sha256": _sha256_file(multiscale_dir / "boundary_provenance.json"),
            "frozen_manifest_hashes_file": str(FROZEN_MANIFEST_HASHES),
            "frozen_manifest_hashes_file_sha256": _sha256_file(FROZEN_MANIFEST_HASHES) if FROZEN_MANIFEST_HASHES.is_file() else None,
            "native_discharge_support_file": str(NATIVE_DISCHARGE_SUPPORT),
            "native_discharge_support_file_sha256": _sha256_file(NATIVE_DISCHARGE_SUPPORT) if NATIVE_DISCHARGE_SUPPORT.is_file() else None,
            "declared_bounds": r["bounds"],
        },
    }


def main(
    split: str, out_path: Optional[Path] = None, *, allow_overwrite: bool = False,
    cli_argv: Optional[list] = None,
) -> int:
    if split not in ALLOWED_SPLITS:
        raise MatchedSupportError(f"split {split!r} is not one of {ALLOWED_SPLITS!r}")

    frozen_manifest_hashes = _parse_frozen_manifest_hashes(FROZEN_MANIFEST_HASHES)
    native_support = _load_native_discharge_support()

    results = []
    for year, d in RESOLVED_D.items():
        for seed in SEEDS:
            control_name = f"sixyear_corrected_forcing_{year}_zero_multiscale_control_retrain_s{seed}"
            multiscale_name = f"sixyear_corrected_forcing_{year}_zero_multiscale_s{seed}"
            entry = compare_fold_seed_pair(
                STAGE_B_OUTPUT / control_name, STAGE_B_OUTPUT / multiscale_name,
                split=split, expected_d=d, expected_seed=int(seed),
                frozen_manifest_hashes=frozen_manifest_hashes,
                native_support=native_support, label=f"{year}/s{seed}", expected_year=year,
            )
            results.append({"fold": year, "seed": seed, "D": d, **entry})

    if out_path is None:
        out_path = Path(
            "/home/arash/Dropbox/HydroPINN paper/results/framework_decade_20261022/improvement_loop_20261023/"
            f"multiscale_history_length_selection/stage_b_matched_support_metrics_{split}.json"
        )
    if out_path.exists() and not allow_overwrite:
        raise MatchedSupportError(
            f"Refusing to overwrite existing output file {out_path} -- pass an explicit new path, "
            "or allow_overwrite=True if regenerating it deliberately (Codex 2026-11-16 item 5)"
        )

    report = {
        "split": split,
        "cli_invocation": " ".join(cli_argv) if cli_argv else f"{Path(__file__).name} {split}",
        "metric_and_unit_definitions": {
            "nse": "Nash-Sutcliffe efficiency, unitless",
            "kge": "Kling-Gupta efficiency, unitless",
            "rmse_mm_h": "RMSE in mm/h (frozen internal training/export unit)",
            "rmse_m3_s": f"RMSE in m^3/s = rmse_mm_h * {CATCHMENT_AREA_KM2} / 3.6",
            "mae_mm_h": "MAE in mm/h",
            "mae_m3_s": f"MAE in m^3/s = mae_mm_h * {CATCHMENT_AREA_KM2} / 3.6",
            "pbias_signed_percent": "100 * sum(predicted-observed) / sum(observed)",
            "pbias_absolute_percent": "abs(pbias_signed_percent) for positive observed-volume sum",
            "catchment_area_km2": CATCHMENT_AREA_KM2,
        },
        "frozen_manifest_hashes_file": str(FROZEN_MANIFEST_HASHES),
        "native_discharge_support_file": str(NATIVE_DISCHARGE_SUPPORT),
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))
    return 0


def _parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "split", nargs="?", default="test", choices=ALLOWED_SPLITS,
        help=f"Which split to score ({'/'.join(ALLOWED_SPLITS)}); defaults to 'test'.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Explicit output JSON path; defaults to the standard per-split path under "
        "multiscale_history_length_selection/.",
    )
    parser.add_argument(
        "--allow-overwrite", action="store_true",
        help="Permit overwriting an existing output file at the resolved path (refused by default).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args(sys.argv[1:])
    raise SystemExit(main(
        _args.split, _args.out, allow_overwrite=_args.allow_overwrite,
        cli_argv=[Path(__file__).name] + sys.argv[1:],
    ))
