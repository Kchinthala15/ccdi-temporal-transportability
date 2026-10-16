#!/usr/bin/env python3
"""Independent TARGET-NBL cohort rebuild and temporal Cox-model refit.

Run from the repository root, where the raw GDC exports are at:
    data/TARGET-NBL/clinical.tsv
    data/TARGET-NBL/follow_up.tsv

The patient-level rebuilt cohort is written only to work/nbl_patient_cohort.csv.
Only aggregate results are printed. The endpoint is the GDC-recorded first-event
field and its recorded time; it should not automatically be interpreted as
identical to a publication-defined event-free survival (EFS) endpoint.

Method reproduced from the independent TARGET-NBL audit:
- one row per case, using the first nonmissing clinical value in export order;
- MYCN amplification and ploidy taken from the first qualifying molecular row
  per case in follow_up.tsv export order;
- first nonmissing recorded first-event/time row per case;
- complete-case cohort requiring age, diagnosis year, an allowed INSS stage,
  binary MYCN result, binary ploidy result, first-event label, and event time;
- age >= 547 days, stage indicators for Stage 3/4/4S, MYCN amplification, and
  diploidy as Cox-model predictors; development <= 2008 and validation 2009-2012;
- CoxPHSurvivalAnalysis with alpha=1e-4, validation calibration slope estimated
  by refitting a Cox model on the validation linear predictor, and 3-year
  observed survival estimated by Kaplan-Meier versus mean model-predicted
  survival within validation age groups.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "TARGET-NBL"
WORK_DIR = ROOT / "work"
CLINICAL_PATH = RAW_DIR / "clinical.tsv"
FOLLOW_UP_PATH = RAW_DIR / "follow_up.tsv"
COHORT_PATH = WORK_DIR / "nbl_patient_cohort.csv"

ID = "cases.submitter_id"
AGE_COL = "diagnoses.age_at_diagnosis"
STAGE_COL = "diagnoses.inss_stage"
YEAR_COL = "diagnoses.year_of_diagnosis"
MYCN_COL = "MYCN"
PLOIDY_COL = "ploidy"
EVENT_COL = "first_event"
TIME_COL = "efs_days"
VALID_STAGES = {"Stage 1", "Stage 2A", "Stage 2B", "Stage 3", "Stage 4", "Stage 4S"}
PREDICTORS = ["age18", "st3", "st4", "st4s", "mycn", "dip"]
HORIZON_DAYS = 3 * 365.25


def require_columns(frame: pd.DataFrame, required: list[str], filename: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing required GDC export columns: {', '.join(missing)}"
        )


def first_nonmissing_by_case(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Keep the first nonmissing value per case in raw export order."""
    return frame[[ID, column]].dropna(subset=[column]).drop_duplicates(ID)


def rebuild_patient_cohort() -> pd.DataFrame:
    if not CLINICAL_PATH.is_file() or not FOLLOW_UP_PATH.is_file():
        raise FileNotFoundError(
            "Could not find both raw GDC files. Expected relative paths: "
            "data/TARGET-NBL/clinical.tsv and data/TARGET-NBL/follow_up.tsv. "
            "Run this script from the repository root."
        )

    clinical = pd.read_csv(CLINICAL_PATH, sep="\t", dtype=str, low_memory=False)
    follow_up = pd.read_csv(FOLLOW_UP_PATH, sep="\t", dtype=str, low_memory=False)
    # The GDC export uses the literal string ' --' / '--' in some missing fields.
    clinical = clinical.replace(["'--", "--"], pd.NA)
    follow_up = follow_up.replace(["'--", "--"], pd.NA)

    clinical_columns = [
        AGE_COL,
        STAGE_COL,
        "diagnoses.mitosis_karyorrhexis_index",
        "diagnoses.inpc_grade",
        "diagnoses.cog_neuroblastoma_risk_group",
        YEAR_COL,
        "demographic.sex_at_birth",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.primary_diagnosis",
    ]
    require_columns(clinical, [ID, *clinical_columns], CLINICAL_PATH)
    molecular_columns = [
        "molecular_tests.molecular_test_id",
        "molecular_tests.gene_symbol",
        "molecular_tests.variant_type",
        "molecular_tests.test_result",
        "molecular_tests.ploidy",
    ]
    event_columns = ["follow_ups.first_event", "follow_ups.days_to_first_event"]
    require_columns(follow_up, [ID, *molecular_columns, *event_columns], FOLLOW_UP_PATH)

    # Reproduce the audited one-row-per-case clinical assembly, preserving export order
    # when selecting the first nonmissing value for a case.
    base = pd.DataFrame({ID: sorted(clinical[ID].dropna().unique())})
    for column in clinical_columns:
        base = base.merge(first_nonmissing_by_case(clinical, column), on=ID, how="left")

    molecular = follow_up[
        follow_up["molecular_tests.molecular_test_id"].notna()
    ].drop_duplicates("molecular_tests.molecular_test_id")

    mycn = molecular[
        (molecular["molecular_tests.gene_symbol"] == "MYCN")
        & (molecular["molecular_tests.variant_type"] == "Amplification")
    ][[ID, "molecular_tests.test_result"]].drop_duplicates(ID).rename(
        columns={"molecular_tests.test_result": MYCN_COL}
    )
    ploidy = molecular[molecular["molecular_tests.ploidy"].notna()][
        [ID, "molecular_tests.ploidy"]
    ].drop_duplicates(ID).rename(columns={"molecular_tests.ploidy": PLOIDY_COL})

    recorded_event = follow_up[follow_up["follow_ups.first_event"].notna()][
        [ID, "follow_ups.first_event", "follow_ups.days_to_first_event"]
    ].drop_duplicates(ID).rename(
        columns={
            "follow_ups.first_event": EVENT_COL,
            "follow_ups.days_to_first_event": TIME_COL,
        }
    )

    cohort = base.merge(mycn, on=ID, how="left")
    cohort = cohort.merge(ploidy, on=ID, how="left")
    cohort = cohort.merge(recorded_event, on=ID, how="left")

    # Save any patient-level artifact only under work/.
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(COHORT_PATH, index=False)
    return cohort


def km_survival_at_horizon(frame: pd.DataFrame) -> float:
    """Return the audit's KM survival estimate at the 3-year horizon."""
    if frame.empty:
        return float("nan")
    event, time = frame["event"].to_numpy(dtype=bool), frame["time"].to_numpy(dtype=float)
    event_times, survival = kaplan_meier_estimator(event, time)
    at_or_before = event_times <= HORIZON_DAYS
    return float(survival[at_or_before][-1]) if at_or_before.any() else 1.0


def mean_predicted_survival_at_horizon(model, frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    functions = model.predict_survival_function(frame[PREDICTORS])
    predictions = np.asarray(
        [function(min(HORIZON_DAYS, function.x[-1])) for function in functions],
        dtype=float,
    )
    return float(np.mean(predictions))


def c_index(frame: pd.DataFrame, linear_predictor: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            frame["event"].to_numpy(dtype=bool),
            frame["time"].to_numpy(dtype=float),
            np.asarray(linear_predictor, dtype=float),
        )[0]
    )


def main() -> int:
    cohort = rebuild_patient_cohort()
    cohort["year"] = pd.to_numeric(cohort[YEAR_COL], errors="coerce")
    cohort["age_days"] = pd.to_numeric(cohort[AGE_COL], errors="coerce")
    cohort["time"] = pd.to_numeric(cohort[TIME_COL], errors="coerce")

    complete_case = (
        cohort["time"].notna()
        & cohort[EVENT_COL].notna()
        & cohort[MYCN_COL].isin(["Amplified", "Not Amplified"])
        & cohort[PLOIDY_COL].isin(["Hyperdiploid", "Diploid"])
        & cohort[STAGE_COL].isin(VALID_STAGES)
        & cohort["age_days"].notna()
        & cohort["year"].notna()
    )
    analytic = cohort.loc[complete_case].copy()
    analytic["event"] = analytic[EVENT_COL].ne("Censored")
    analytic["time"] = analytic["time"].clip(lower=1)
    analytic["age18"] = (analytic["age_days"] >= 547).astype(int)
    analytic["st3"] = analytic[STAGE_COL].eq("Stage 3").astype(int)
    analytic["st4"] = analytic[STAGE_COL].eq("Stage 4").astype(int)
    analytic["st4s"] = analytic[STAGE_COL].eq("Stage 4S").astype(int)
    analytic["mycn"] = analytic[MYCN_COL].eq("Amplified").astype(int)
    analytic["dip"] = analytic[PLOIDY_COL].eq("Diploid").astype(int)

    development = analytic.loc[analytic["year"] <= 2008].copy()
    validation = analytic.loc[analytic["year"].between(2009, 2012, inclusive="both")].copy()
    outside_periods = len(analytic) - len(development) - len(validation)

    if development.empty or validation.empty:
        raise ValueError(
            f"Development or validation cohort is empty (development={len(development)}, "
            f"validation={len(validation)}). Check the source export and diagnosis years."
        )

    survival_development = Surv.from_arrays(
        development["event"].to_numpy(dtype=bool),
        development["time"].to_numpy(dtype=float),
    )
    model = CoxPHSurvivalAnalysis(alpha=1e-4).fit(
        development[PREDICTORS], survival_development
    )

    lp_development = model.predict(development[PREDICTORS])
    lp_validation = model.predict(validation[PREDICTORS])
    validation_survival = Surv.from_arrays(
        validation["event"].to_numpy(dtype=bool),
        validation["time"].to_numpy(dtype=float),
    )
    calibration_model = CoxPHSurvivalAnalysis(alpha=1e-6).fit(
        np.asarray(lp_validation).reshape(-1, 1), validation_survival
    )
    calibration_slope = float(calibration_model.coef_[0])

    print("TARGET-NBL independent rebuild (aggregate results only)")
    print(f"Rebuilt patient records: {len(cohort)}")
    print(
        f"Complete-case analytic cohort: n={len(analytic)}, "
        f"recorded events={int(analytic['event'].sum())}, "
        f"censored={int((~analytic['event']).sum())}"
    )
    print(
        f"Development (diagnosis year <= 2008): n={len(development)}, "
        f"recorded events={int(development['event'].sum())}, "
        f"censored={int((~development['event']).sum())}"
    )
    print(
        f"Validation (diagnosis years 2009-2012): n={len(validation)}, "
        f"recorded events={int(validation['event'].sum())}, "
        f"censored={int((~validation['event']).sum())}"
    )
    if outside_periods:
        print(f"Analytic-cohort records outside the specified development/validation eras: {outside_periods}")
    print(f"Cox C-index in development: {c_index(development, lp_development):.4f}")
    print(f"Cox C-index in validation: {c_index(validation, lp_validation):.4f}")
    print(f"Validation calibration slope: {calibration_slope:.4f}")

    validation = validation.copy()
    validation["predicted_survival_3y"] = np.asarray(
        [
            function(min(HORIZON_DAYS, function.x[-1]))
            for function in model.predict_survival_function(validation[PREDICTORS])
        ],
        dtype=float,
    )
    print("Validation 3-year recorded-EFS O-E by age group (Kaplan-Meier observed minus mean predicted; percentage points):")
    for label, mask in [
        ("Age < 18 months", validation["age18"].eq(0)),
        ("Age >= 18 months", validation["age18"].eq(1)),
    ]:
        subgroup = validation.loc[mask]
        if subgroup.empty:
            print(f"  {label}: n=0; estimate unavailable")
            continue
        observed = km_survival_at_horizon(subgroup)
        predicted = float(subgroup["predicted_survival_3y"].mean())
        difference_pp = (observed - predicted) * 100
        print(
            f"  {label}: n={len(subgroup)}, recorded events={int(subgroup['event'].sum())}, "
            f"observed KM={observed:.4f}, mean predicted={predicted:.4f}, "
            f"O-E={difference_pp:+.2f} pp"
        )

    print(f"Patient-level rebuilt cohort saved under work/: {COHORT_PATH.relative_to(ROOT)}")
    print("Endpoint caveat: these are GDC-recorded first-event results, not necessarily a publication-identical formal EFS endpoint.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
