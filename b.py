from __future__ import annotations

import json
import os
import re
import random
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You convert structured clinical data into two educational video prompts for a
future video-language model.

Return ONLY valid JSON as an array. Each array item must contain:
- patient_id
- base_video_prompt
- detailed_mechanism_prompt

Rules:
1.  Use only the clinical facts provided.
2.  Keep demographics brief and essential only.
3.  Mention the main diagnosis and the most relevant symptoms/findings.
4.  Mention the prescribed drug, dose, route, frequency, and treatment duration.
5.  For base_video_prompt, include the exact phrase:
    "generate a video animation that shows how this drug works in the human body"
6.  For detailed_mechanism_prompt, explicitly request:
    - how the drug is taken
    - how it travels through the body
    - the target organ or site of action
    - a high-level mechanism of action
    - clear labels for organs, tissues, and the drug along its path
7.  Keep prompts specific to the diagnosis and drug.
8.  Do not invent extra diagnoses, contraindications, or side effects.
9.  If a field is missing, say "not specified" rather than guessing.
10. Output English only.
"""

_EXPECTED_PATIENT_COLS = ["ID", "BIRTHDATE", "GENDER", "ETHNICITY"]
_EXPECTED_CONDITION_COLS = ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"]
_EXPECTED_MEDICATION_COLS = [
    "START", "STOP", "PATIENT", "CODE", "DESCRIPTION",
    "REASONCODE", "REASONDESCRIPTION",
]
_EXPECTED_OBSERVATION_COLS = ["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS"]
_EXPECTED_SYMPTOM_COLS = ["PATIENT", "PATHOLOGY", "SYMPTOMS", "NUM_SYMPTOMS"]

# ---------------------------------------------------------------------------
# Helpers – column normalisation & safe access
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip().upper().replace(" ", "_").replace("/", "_")
        for c in df.columns
    ]
    return df


def _ensure_columns(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    return df


def _col(row: pd.Series, name: str, default: str = "") -> str:
    val = row.get(name, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip()


# ---------------------------------------------------------------------------
# Helpers – parsing & formatting
# ---------------------------------------------------------------------------

def _parse_dt(x: str) -> Any:
    """Parse a date/datetime string; return pd.NaT on failure (tz-naive)."""
    if not x or str(x).strip() == "":
        return pd.NaT
    ts = pd.to_datetime(x, errors="coerce")
    if ts is pd.NaT:
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _parse_dt_series(s: pd.Series) -> pd.Series:
    """Vectorised tz-naive datetime parsing for an entire Series."""
    parsed = pd.to_datetime(s, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


def _clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).strip().split())


def _first(*vals: Any, default: str = "not specified") -> str:
    for v in vals:
        s = _clean(v)
        if s:
            return s
    return default


def _age_from_birthdate(raw: str) -> Optional[int]:
    try:
        b = datetime.strptime(raw[:10], "%Y-%m-%d").date()
        t = date.today()
        return t.year - b.year - ((t.month, t.day) < (b.month, b.day))
    except Exception:
        return None


def _gender_text(raw: str) -> str:
    g = _clean(raw).upper()
    return {"M": "male", "F": "female"}.get(g, g.lower() if g else "not specified")


def _normalise_drug_name(name: str) -> str:
    return re.sub(r"\s+", " ", _clean(name).lower())


def _is_missing(x: Optional[str]) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"not specified", "none", "null", "nan"}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_csv_files(root: Path, csv_name: str) -> List[Path]:
    found: List[Path] = []
    std = root / "csv" / csv_name
    if std.is_file():
        found.append(std)
    found.extend(sorted(root.glob(f"run_*/csv/{csv_name}")))
    return found


def _discover_symptom_files(root: Path) -> List[Path]:
    found: List[Path] = []
    found.extend(
        sorted((root / "symptoms" / "csv").glob("*.csv"))
        if (root / "symptoms" / "csv").is_dir() else []
    )
    found.extend(sorted(root.glob("run_*/symptoms/csv/*.csv")))
    return found


def _discover_fhir_bundles(root: Path) -> List[Path]:
    found: List[Path] = []
    fhir_dir = root / "fhir"
    if fhir_dir.is_dir():
        found.extend(sorted(fhir_dir.rglob("*.json")))
    for run_dir in sorted(root.glob("run_*")):
        rd = run_dir / "fhir"
        if rd.is_dir():
            found.extend(sorted(rd.rglob("*.json")))
    return found


# ---------------------------------------------------------------------------
# CSV readers  (OPT: only load needed columns; concat once)
# ---------------------------------------------------------------------------

def _read_csvs(
    paths: List[Path],
    expected_cols: List[str],
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read and concatenate CSVs.  Pass *usecols* to skip unneeded columns."""
    dfs: List[pd.DataFrame] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            # Peek at the header to figure out which usecols actually exist.
            if usecols:
                header = pd.read_csv(p, nrows=0)
                header.columns = [
                    c.strip().upper().replace(" ", "_").replace("/", "_")
                    for c in header.columns
                ]
                available = [c for c in usecols if c in header.columns]
                # Map back to original column names for pd.read_csv
                orig_cols = {
                    c.strip().upper().replace(" ", "_").replace("/", "_"): c
                    for c in pd.read_csv(p, nrows=0).columns
                }
                read_cols = [orig_cols[c] for c in available if c in orig_cols]
                df = pd.read_csv(p, dtype=str, usecols=read_cols or None,
                                 low_memory=False).fillna("")
            else:
                df = pd.read_csv(p, dtype=str, low_memory=False).fillna("")
            df = _normalise_columns(df)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return _ensure_columns(pd.DataFrame(), expected_cols)
    merged = pd.concat(dfs, ignore_index=True)
    return _ensure_columns(merged, expected_cols)


def _read_symptoms(root: Path) -> pd.DataFrame:
    paths = _discover_symptom_files(root)
    return _read_csvs(paths, _EXPECTED_SYMPTOM_COLS)


# ---------------------------------------------------------------------------
# Symptom text parsing
# ---------------------------------------------------------------------------

def _parse_symptom_text(raw: str) -> List[str]:
    if not raw or not raw.strip():
        return []
    parts = raw.split(";")
    out: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        name = part.split(":")[0].strip()
        if name and not name.replace(".", "").replace("-", "").isdigit():
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# FHIR bundle helpers
# ---------------------------------------------------------------------------

def _coding_display(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    text = _clean(obj.get("text"))
    if text:
        return text
    for c in obj.get("coding", []) or []:
        display = _clean(c.get("display"))
        if display:
            return display
        code = _clean(c.get("code"))
        if code:
            return code
    return ""


def _coding_code(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    for c in obj.get("coding", []) or []:
        code = _clean(c.get("code"))
        if code:
            return code
    return ""


def _dose_text(di: dict) -> str:
    if not isinstance(di, dict):
        return ""
    for dr in di.get("doseAndRate", []) or []:
        dq = dr.get("doseQuantity", {})
        value = _clean(dq.get("value"))
        unit = _clean(dq.get("unit"))
        if value or unit:
            return f"{value} {unit}".strip()
    return ""


def _stringify_repeat(rep: dict) -> str:
    if not isinstance(rep, dict):
        return ""
    freq = rep.get("frequency")
    period = rep.get("period")
    unit = _clean(rep.get("periodUnit"))
    when = rep.get("when")
    parts: List[str] = []
    if freq and period and unit:
        parts.append(f"{freq} times every {period} {unit}")
    elif freq:
        parts.append(f"{freq} times")
    if when:
        if isinstance(when, list):
            parts.append(" ".join(map(str, when)))
        else:
            parts.append(str(when))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# FHIR bundle extraction  (OPT: parallel bundle parsing)
# ---------------------------------------------------------------------------

def _parse_single_bundle(fp: Path) -> List[Dict[str, str]]:
    """Parse one FHIR bundle file; return a list of medication rows."""
    try:
        bundle = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        return []

    entries = bundle.get("entry", []) or []
    patient_id: Optional[str] = None
    resource_index: Dict[str, dict] = {}
    resources: List[dict] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        res = entry.get("resource")
        if not isinstance(res, dict):
            continue
        resources.append(res)
        full_url = entry.get("fullUrl", "")
        if full_url:
            resource_index[full_url] = res
        rid = res.get("id", "")
        rtype = res.get("resourceType", "")
        if rid and rtype:
            resource_index[f"{rtype}/{rid}"] = res
        if rtype == "Patient" and not patient_id:
            patient_id = _clean(rid)

    if not patient_id:
        patient_id = fp.stem

    rows: List[Dict[str, str]] = []
    for res in resources:
        if res.get("resourceType") != "MedicationRequest":
            continue

        med = res.get("medicationCodeableConcept", {})
        med_name = _coding_display(med)

        reason = ""
        reason_code = ""
        rcodes = res.get("reasonCode", []) or []
        if rcodes and isinstance(rcodes, list) and isinstance(rcodes[0], dict):
            reason = _coding_display(rcodes[0])
            reason_code = _coding_code(rcodes[0])

        if not reason and not reason_code:
            refs = res.get("reasonReference", []) or []
            for ref in (refs if isinstance(refs, list) else [refs]):
                if not isinstance(ref, dict):
                    continue
                ref_str = ref.get("reference", "")
                target = resource_index.get(ref_str)
                if target and target.get("resourceType") == "Condition":
                    code_obj = target.get("code", {})
                    reason = _coding_display(code_obj)
                    reason_code = _coding_code(code_obj)
                    if reason or reason_code:
                        break

        di = {}
        dosage_instructions = res.get("dosageInstruction", []) or []
        if dosage_instructions and isinstance(dosage_instructions[0], dict):
            di = dosage_instructions[0]

        dosage = _first(_dose_text(di), _clean(di.get("text")), default="not specified")
        route = _first(_coding_display(di.get("route", {})), default="not specified")
        frequency = _first(
            _clean(di.get("text")),
            _stringify_repeat(di.get("timing", {}).get("repeat", {})),
            default="not specified",
        )

        validity = (res.get("dispenseRequest") or {}).get("validityPeriod") or {}
        authored_on = _clean(res.get("authoredOn"))
        start = _first(authored_on, _clean(validity.get("start")), default="")
        stop = _first(_clean(validity.get("end")), default="")
        duration = "not specified"
        if start and stop:
            try:
                s = _parse_dt(start)
                e = _parse_dt(stop)
                if pd.notna(s) and pd.notna(e):
                    duration = f"{max((e - s).days, 0)} days"
            except Exception:
                pass
        elif start:
            duration = "ongoing"

        rows.append({
            "patient_id": patient_id,
            "drug_name_fhir": _first(med_name, default=""),
            "dosage_fhir": dosage,
            "route_fhir": route,
            "frequency_fhir": frequency,
            "duration_fhir": duration,
            "start_fhir": start,
            "stop_fhir": stop,
            "reason_code_fhir": reason_code,
            "reason_description_fhir": reason,
        })
    return rows


_FHIR_EMPTY_COLS = [
    "patient_id", "drug_name_fhir", "dosage_fhir", "route_fhir",
    "frequency_fhir", "duration_fhir", "start_fhir", "stop_fhir",
    "reason_code_fhir", "reason_description_fhir",
]


def _extract_fhir_data(root: Path, max_workers: int = 8) -> pd.DataFrame:
    """Parse all FHIR bundles in parallel and return a combined DataFrame."""
    bundle_paths = _discover_fhir_bundles(root)
    if not bundle_paths:
        return pd.DataFrame(columns=_FHIR_EMPTY_COLS)

    all_rows: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_parse_single_bundle, fp): fp for fp in bundle_paths}
        for fut in as_completed(futures):
            try:
                all_rows.extend(fut.result())
            except Exception:
                pass

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=_FHIR_EMPTY_COLS)


# ---------------------------------------------------------------------------
# Clinical logic – choose main condition  (OPT: vectorised date sort)
# ---------------------------------------------------------------------------

def _choose_main_condition(
    cond_df: pd.DataFrame, med_df: pd.DataFrame
) -> Optional[Dict[str, str]]:
    if cond_df.empty:
        return None

    temp = cond_df.copy()
    temp["_start_dt"] = _parse_dt_series(temp["START"])

    active = temp[temp["STOP"].eq("")]
    pool = active if not active.empty else temp
    pool = pool.sort_values("_start_dt", ascending=False, na_position="last")

    if not active.empty and not med_df.empty:
        med_reason_codes = set(med_df["REASONCODE"].str.strip().str.lower()) - {""}
        linked = pool[pool["CODE"].str.strip().str.lower().isin(med_reason_codes)]
        if not linked.empty:
            pool = linked

    row = pool.iloc[0]
    return {
        "code": _first(_col(row, "CODE"), default="not specified"),
        "name": _first(_col(row, "DESCRIPTION"), default="not specified"),
        "severity": "not specified",
        "onset": _first(_col(row, "START"), default="not specified"),
    }


# ---------------------------------------------------------------------------
# Clinical logic – symptoms & findings  (OPT: vectorised date sort)
# ---------------------------------------------------------------------------

def _choose_symptoms(
    patient_id: str,
    dx_name: str,
    dx_onset: str,
    sym_df: pd.DataFrame,
    obs_df: pd.DataFrame,
) -> List[str]:
    out: List[str] = []

    if not sym_df.empty:
        psym = sym_df[sym_df["PATIENT"] == patient_id].copy()
        if not psym.empty:
            if dx_name and dx_name != "not specified":
                match = psym[psym["PATHOLOGY"].str.lower() == dx_name.lower()]
                if not match.empty:
                    psym = match
            for _, srow in psym.head(5).iterrows():
                out.extend(_parse_symptom_text(_col(srow, "SYMPTOMS")))

    if len(out) < 5 and not obs_df.empty:
        pobs = obs_df[obs_df["PATIENT"] == patient_id].copy()
        if not pobs.empty:
            pobs["_dt"] = _parse_dt_series(pobs["DATE"])
            onset_dt = _parse_dt(dx_onset)
            if pd.notna(onset_dt):
                pobs["_dist"] = (pobs["_dt"] - onset_dt).abs()
                pobs = pobs.sort_values("_dist", na_position="last")
            else:
                pobs = pobs.sort_values("_dt", ascending=False, na_position="last")
            for _, orow in pobs.head(5).iterrows():
                desc = _col(orow, "DESCRIPTION")
                val = _col(orow, "VALUE")
                unit = _col(orow, "UNITS")
                if not desc:
                    continue
                txt = desc
                if val:
                    txt += f": {val}"
                    if unit:
                        txt += f" {unit}"
                out.append(txt)

    seen: set[str] = set()
    dedup: List[str] = []
    for s in out:
        key = s.lower()
        if key not in seen:
            dedup.append(s)
            seen.add(key)

    return dedup[:5] if dedup else ["not specified"]


# ---------------------------------------------------------------------------
# Clinical logic – medication selection  (OPT: vectorised date sort)
# ---------------------------------------------------------------------------

def _has_complete_prescription(rx: Dict[str, str]) -> bool:
    if not rx:
        return False
    if _is_missing(rx.get("drug_name")):
        return False
    return any(not _is_missing(rx.get(k)) for k in ("dosage", "frequency", "start"))


def _choose_medication(
    patient_id: str,
    dx: Dict[str, str],
    pmeds: pd.DataFrame,          # already filtered to this patient
    fhir_rx_df: pd.DataFrame,
) -> Optional[Dict[str, str]]:
    csv_drug_name = csv_start = csv_stop = csv_reason_code = csv_reason_description = ""

    if not pmeds.empty:
        pmeds = pmeds.copy()
        pmeds["_sort_dt"] = _parse_dt_series(pmeds["START"])

        dx_name_lower = dx["name"].lower()
        dx_code_lower = dx["code"].lower()
        pmeds["_match"] = 0
        if dx_name_lower and dx_name_lower != "not specified":
            pmeds.loc[
                pmeds["REASONDESCRIPTION"].str.strip().str.lower() == dx_name_lower, "_match"
            ] += 2
        if dx_code_lower and dx_code_lower != "not specified":
            pmeds.loc[
                pmeds["REASONCODE"].str.strip().str.lower() == dx_code_lower, "_match"
            ] += 3

        pmeds = pmeds.sort_values(
            ["_match", "_sort_dt"], ascending=[False, False], na_position="last"
        )
        best = pmeds.iloc[0]
        csv_drug_name       = _col(best, "DESCRIPTION")
        csv_start           = _col(best, "START")
        csv_stop            = _col(best, "STOP")
        csv_reason_code     = _col(best, "REASONCODE")
        csv_reason_description = _col(best, "REASONDESCRIPTION")

    fhir_cands = fhir_rx_df.get(patient_id, pd.DataFrame(columns=_FHIR_EMPTY_COLS))

    if fhir_cands.empty and not csv_drug_name:
        return None

    if not fhir_cands.empty:
        fhir_cands = fhir_cands.copy()
        if csv_drug_name:
            csv_norm = _normalise_drug_name(csv_drug_name)
            fhir_cands["_name_match"] = fhir_cands["drug_name_fhir"].apply(
                lambda x: (
                    1 if csv_norm in _normalise_drug_name(x)
                    or _normalise_drug_name(x) in csv_norm
                    else 0
                )
            )
        else:
            fhir_cands["_name_match"] = 0

        dx_code_lower = dx["code"].lower()
        dx_name_lower = dx["name"].lower()
        fhir_cands["_reason_match"] = 0
        if dx_code_lower and dx_code_lower != "not specified":
            fhir_cands.loc[
                fhir_cands["reason_code_fhir"].str.strip().str.lower() == dx_code_lower,
                "_reason_match",
            ] += 3
        if dx_name_lower and dx_name_lower != "not specified":
            fhir_cands.loc[
                fhir_cands["reason_description_fhir"].str.strip().str.lower() == dx_name_lower,
                "_reason_match",
            ] += 2

        detail_fields = ["dosage_fhir", "route_fhir", "frequency_fhir", "duration_fhir"]
        fhir_cands["_completeness"] = fhir_cands[detail_fields].apply(
            lambda r: sum(1 for v in r if not _is_missing(v)), axis=1
        )
        fhir_cands["_start_dt"] = _parse_dt_series(fhir_cands["start_fhir"])
        fhir_cands = fhir_cands.sort_values(
            ["_reason_match", "_name_match", "_completeness", "_start_dt"],
            ascending=[False, False, False, False],
            na_position="last",
        )

        for _, fr in fhir_cands.iterrows():
            candidate = {
                "drug_name":            _first(fr.get("drug_name_fhir", ""), csv_drug_name),
                "dosage":               _first(fr.get("dosage_fhir", ""), default="not specified"),
                "route":                _first(fr.get("route_fhir", ""), default="not specified"),
                "frequency":            _first(fr.get("frequency_fhir", ""), default="not specified"),
                "duration":             _first(fr.get("duration_fhir", ""), default="not specified"),
                "start":                _first(fr.get("start_fhir", ""), default=""),
                "stop":                 _first(fr.get("stop_fhir", ""), default=""),
                "reason_code":          _first(fr.get("reason_code_fhir", ""), csv_reason_code, default=""),
                "reason_description":   _first(fr.get("reason_description_fhir", ""), csv_reason_description, default=""),
            }
            if _has_complete_prescription(candidate):
                return candidate

    if csv_drug_name:
        csv_duration = "not specified"
        if csv_start and csv_stop:
            try:
                s = _parse_dt(csv_start)
                e = _parse_dt(csv_stop)
                if pd.notna(s) and pd.notna(e):
                    csv_duration = f"{max((e - s).days, 0)} days"
            except Exception:
                pass
        elif csv_start:
            csv_duration = "ongoing"
        candidate = {
            "drug_name":            csv_drug_name,
            "dosage":               "not specified",
            "route":                "not specified",
            "frequency":            "not specified",
            "duration":             csv_duration,
            "start":                csv_start,
            "stop":                 csv_stop,
            "reason_code":          csv_reason_code,
            "reason_description":   csv_reason_description,
        }
        if _has_complete_prescription(candidate):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Completeness checks
# ---------------------------------------------------------------------------

def _has_complete_diagnosis(dx: Optional[Dict[str, str]]) -> bool:
    if not dx:
        return False
    return not _is_missing(dx.get("code")) or not _is_missing(dx.get("name"))



def _is_complete_record(record: Dict) -> bool:
    if not _has_complete_diagnosis(record.get("main_diagnosis")):
        return False
    if not _has_complete_prescription(record.get("prescription")):
        return False
    return True


# _STRICT_DEMOGRAPHICS_FIELDS   = ("sex", "ethnicity")
# _STRICT_DIAGNOSIS_FIELDS       = ("code", "name", "onset")
# _STRICT_PRESCRIPTION_FIELDS    = ("drug_name", "dosage", "route", "frequency", "duration")
# _STRICT_PRESCRIPTION_REASON_FIELDS = ("reason_code", "reason_description")

_STRICT_DEMOGRAPHICS_FIELDS = ("age", "sex")
_STRICT_DIAGNOSIS_FIELDS = ("code", "name", "onset")
_STRICT_PRESCRIPTION_FIELDS = ("drug_name",)

# Make reason fields optional
_STRICT_PRESCRIPTION_REASON_FIELDS = ()

def _is_strictly_complete(record: Dict) -> bool:
    if _is_missing(record.get("patient_id")):
        return False
    demo = record.get("brief_demographics") or {}
    for field in _STRICT_DEMOGRAPHICS_FIELDS:
        val = demo.get(field)
        if _is_missing(val if isinstance(val, str) else (None if val is None else str(val))):
            return False
    dx = record.get("main_diagnosis") or {}
    for field in _STRICT_DIAGNOSIS_FIELDS:
        if _is_missing(dx.get(field)):
            return False
    sx = record.get("symptoms_and_findings") or []
    if not sx or all(_is_missing(s) for s in sx):
        return False
    rx = record.get("prescription") or {}
    for field in _STRICT_PRESCRIPTION_FIELDS:
        if _is_missing(rx.get(field)):
            return False
    for field in _STRICT_PRESCRIPTION_REASON_FIELDS:
        if field in rx and _is_missing(rx[field]):
            return False
    return True


# ---------------------------------------------------------------------------
# Record building  (OPT: pre-group DataFrames once; pre-index FHIR by patient)
# ---------------------------------------------------------------------------

def build_records(root: Path, max_fhir_workers: int = 8) -> List[Dict]:
    """Build structured patient records from Synthea exports at *root*."""
    patients   = _read_csvs(_discover_csv_files(root, "patients.csv"),    _EXPECTED_PATIENT_COLS)
    conditions = _read_csvs(_discover_csv_files(root, "conditions.csv"),  _EXPECTED_CONDITION_COLS)
    medications= _read_csvs(_discover_csv_files(root, "medications.csv"), _EXPECTED_MEDICATION_COLS)
    observations=_read_csvs(_discover_csv_files(root, "observations.csv"),_EXPECTED_OBSERVATION_COLS)
    symptoms   = _read_symptoms(root)
    fhir_rx    = _extract_fhir_data(root, max_workers=max_fhir_workers)

    if patients.empty:
        raise RuntimeError(
            f"No patients.csv found under {root}. "
            "Expected <root>/csv/patients.csv or <root>/run_*/csv/patients.csv."
        )

    # OPT: Pre-group all per-patient DataFrames once (O(N)) instead of
    #      re-scanning full frames per patient (O(N*M)).
    cond_by_patient = dict(tuple(conditions.groupby("PATIENT"))) if not conditions.empty else {}
    meds_by_patient = dict(tuple(medications.groupby("PATIENT"))) if not medications.empty else {}
    sym_by_patient  = dict(tuple(symptoms.groupby("PATIENT")))    if not symptoms.empty   else {}
    obs_by_patient  = dict(tuple(observations.groupby("PATIENT")))if not observations.empty else {}

    # OPT: Pre-index FHIR rows by patient_id for O(1) lookup.
    fhir_by_patient: Dict[str, pd.DataFrame] = (
        dict(tuple(fhir_rx.groupby("patient_id"))) if not fhir_rx.empty else {}
    )

    _EMPTY_COND = _ensure_columns(pd.DataFrame(), _EXPECTED_CONDITION_COLS)
    _EMPTY_MEDS = _ensure_columns(pd.DataFrame(), _EXPECTED_MEDICATION_COLS)
    _EMPTY_SYM  = _ensure_columns(pd.DataFrame(), _EXPECTED_SYMPTOM_COLS)
    _EMPTY_OBS  = _ensure_columns(pd.DataFrame(), _EXPECTED_OBSERVATION_COLS)
    _EMPTY_FHIR = pd.DataFrame(columns=_FHIR_EMPTY_COLS)

    records: List[Dict] = []
    for _, prow in patients.iterrows():
        patient_id = _col(prow, "ID")
        if not patient_id:
            continue

        pcond = cond_by_patient.get(patient_id, _EMPTY_COND)
        pmeds = meds_by_patient.get(patient_id, _EMPTY_MEDS)

        dx = _choose_main_condition(pcond, pmeds)
        if not dx:
            continue

        rx = _choose_medication(patient_id, dx, pmeds, fhir_by_patient)
        if not rx:
            continue

        sx = _choose_symptoms(
            patient_id,
            dx["name"],
            dx.get("onset", ""),
            sym_by_patient.get(patient_id, _EMPTY_SYM),
            obs_by_patient.get(patient_id, _EMPTY_OBS),
        )
        age = _age_from_birthdate(_col(prow, "BIRTHDATE"))

        record = {
            "patient_id": patient_id,
            "brief_demographics": {
                "age":       age if age is not None else "not specified",
                "sex":       _gender_text(_col(prow, "GENDER")),
                "ethnicity": _first(_col(prow, "ETHNICITY"), default="not specified"),
            },
            "main_diagnosis":        dx,
            "symptoms_and_findings": sx,
            "prescription":          rx,
        }
        if _is_complete_record(record) and _is_strictly_complete(record):
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Diversity sampling
# ---------------------------------------------------------------------------

def _diversity_key(r: Dict) -> str:
    return " | ".join([
        r["main_diagnosis"]["name"],
        r["prescription"]["drug_name"],
        r["prescription"].get("route", "not specified"),
    ])


def _diverse_sample(records: List[Dict], target_n: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict]] = {}
    for r in records:
        buckets.setdefault(_diversity_key(r), []).append(r)
    for v in buckets.values():
        rng.shuffle(v)
    keys = list(buckets.keys())
    rng.shuffle(keys)

    chosen: List[Dict] = []
    while len(chosen) < min(target_n, len(records)):
        progressed = False
        for k in keys:
            if buckets[k]:
                chosen.append(buckets[k].pop())
                progressed = True
                if len(chosen) >= target_n:
                    break
        if not progressed:
            break
    return chosen


# ---------------------------------------------------------------------------
# Gemini integration  (OPT: concurrent batch API calls)
# ---------------------------------------------------------------------------

def _call_gemini(client: Any, model: str, batch: List[Dict]) -> List[Dict]:
    payload = {
        "task": "Generate two educational video prompts per patient record.",
        "records": batch,
    }
    resp = client.models.generate_content(
        model=model,
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )
    data = json.loads(resp.text)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError("Gemini response was not a JSON array")
    return [
        {
            "patient_id":                 str(item.get("patient_id", "")),
            "base_video_prompt":          _clean(item.get("base_video_prompt", "")),
            "detailed_mechanism_prompt":  _clean(item.get("detailed_mechanism_prompt", "")),
        }
        for item in data
    ]


def _call_gemini_with_retry(
    client: Any, model: str, batch: List[Dict], max_retries: int = 4
) -> List[Dict]:
    """Call Gemini with exponential-backoff retry; raise on final failure."""
    for attempt in range(max_retries):
        try:
            return _call_gemini(client, model, batch)
        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"ERROR: Gemini call failed after {max_retries} attempts: {exc}")
                raise
            wait = 2 ** attempt
            print(f"  Gemini attempt {attempt + 1} failed, retrying in {wait}s …")
            time.sleep(wait)
    return []  # unreachable


def _call_gemini_batches_concurrent(
    client: Any,
    model: str,
    batches: List[List[Dict]],
    max_api_workers: int = 4,
) -> List[Dict]:
    """Submit all batches concurrently (up to *max_api_workers* in flight)."""
    prompt_rows: List[Dict] = [None] * len(batches)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=max_api_workers) as pool:
        futures = {
            pool.submit(_call_gemini_with_retry, client, model, batch): idx
            for idx, batch in enumerate(batches)
        }
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            prompt_rows[idx] = fut.result()
            completed += 1
            if completed % 25 == 0:
                print(f"  Processed {completed}/{len(batches)} batches")

    return [row for batch_rows in prompt_rows for row in batch_rows]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _merge_output(records: List[Dict], prompts: List[Dict]) -> pd.DataFrame:
    pmap = {x["patient_id"]: x for x in prompts}
    rows: List[Dict] = []
    for r in records:
        pid = r["patient_id"]
        p = pmap.get(pid, {})
        rows.append({
            "patient_id":                   pid,
            "age":                          r["brief_demographics"]["age"],
            "sex":                          r["brief_demographics"]["sex"],
            "ethnicity":                    r["brief_demographics"]["ethnicity"],
            "diagnosis_code":               r["main_diagnosis"]["code"],
            "diagnosis_name":               r["main_diagnosis"]["name"],
            "diagnosis_severity":           r["main_diagnosis"]["severity"],
            "diagnosis_onset":              r["main_diagnosis"]["onset"],
            "symptoms_and_findings":        " | ".join(r["symptoms_and_findings"]),
            "drug_name":                    r["prescription"]["drug_name"],
            "dosage":                       r["prescription"]["dosage"],
            "route":                        r["prescription"]["route"],
            "frequency":                    r["prescription"]["frequency"],
            "treatment_duration":           r["prescription"]["duration"],
            "prescription_reason_code":     r["prescription"]["reason_code"],
            "prescription_reason_description": r["prescription"]["reason_description"],
            "base_video_prompt":            p.get("base_video_prompt", ""),
            "detailed_mechanism_prompt":    p.get("detailed_mechanism_prompt", ""),
            "structured_record_json":       json.dumps(r, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Gemini video prompts from Synthea patient exports."
    )
    ap.add_argument(
        "--root_dir", required=True,
        help=(
            "Root directory of Synthea output.  Supports standard layout "
            "(csv/, fhir/, symptoms/csv/) and multi-run layout (run_*/csv/, etc.)."
        ),
    )
    ap.add_argument("--runs_dir", dest="root_dir", help=argparse.SUPPRESS)
    ap.add_argument("--target_patients",  type=int, default=5)
    ap.add_argument("--batch_size",       type=int, default=5)
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--model",            default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    ap.add_argument("--out_csv",          default="synthea_video_prompts.csv")
    # OPT: expose worker counts as CLI flags
    ap.add_argument(
        "--fhir_workers", type=int, default=8,
        help="Parallel threads for FHIR bundle parsing (default: 8).",
    )
    ap.add_argument(
        "--api_workers", type=int, default=4,
        help="Concurrent Gemini API calls in flight (default: 4).",
    )
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GOOGLE_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    root = Path(args.root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    print(f"Scanning {root} for Synthea exports …")
    records = build_records(root, max_fhir_workers=args.fhir_workers)
    print(f"Built {len(records)} complete patient records.")

    if len(records) < args.target_patients:
        raise RuntimeError(
            f"Need {args.target_patients} strictly complete records but found only "
            f"{len(records)} after filtering.  Generate more Synthea data and re-run."
        )

    selected = _diverse_sample(records, args.target_patients, seed=args.seed)
    print(f"Selected {len(selected)} diverse patients for prompt generation.")

    batches = [selected[i: i + args.batch_size] for i in range(0, len(selected), args.batch_size)]
    print(f"Submitting {len(batches)} batches (up to {args.api_workers} concurrent) …")

    # OPT: concurrent Gemini API calls
    prompt_rows = _call_gemini_batches_concurrent(client, args.model, batches, args.api_workers)

    final_df = _merge_output(selected, prompt_rows)
    final_df = final_df[
        final_df["base_video_prompt"].str.len().gt(0)
        & final_df["detailed_mechanism_prompt"].str.len().gt(0)
    ].copy()

    final_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(final_df)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
