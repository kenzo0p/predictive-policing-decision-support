"""
Flexible data pipeline with schema validation for crime datasets.

This module standardizes raw datasets into a consistent schema:
    - state
    - district
    - year
    - total_crimes
    - population
    - crime_rate

It supports:
    - automatic column inference from common aliases
    - optional explicit column mapping via JSON
    - basic validation and cleaning
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd


RAW_REQUIRED_COLUMNS = ["state", "district", "year", "total_crimes"]
FINAL_COLUMNS = ["state", "year", "total_crimes", "population", "crime_rate"]

DEFAULT_ALIASES = {
    "state": [
        "STATE/UT", "STATE", "STATE UT", "STATE_UT", "STATEUT",
        "STATE/UNION TERRITORY", "STATE/UT NAME", "STATE NAME"
    ],
    "district": [
        "DISTRICT", "DIST", "DISTRICT NAME", "DISTRICT_NAME",
        "DISTRICT/UT", "DISTRICT UT"
    ],
    "year": ["YEAR", "YR", "CRIME_YEAR"],
    "total_crimes": [
        "TOTAL IPC CRIMES", "TOTAL_IPC_CRIMES", "TOTAL IPC",
        "TOTAL CRIMES", "TOTAL_CRIMES", "TOTAL"
    ],
    "population": [
        "POPULATION", "TOTAL POPULATION", "POP", "POPN", "STATE POPULATION"
    ],
}

AGGREGATE_DISTRICT_PATTERN = r"TOTAL|RLY|G\.R\.P|CID|STF|BIEO|R\.P\.O"

STATE_POPULATION_2011 = {
    "ANDISLANDS": 380581,
    "ANDHRAPRADESH": 84580777,
    "ARUNACHALPRADESH": 1383727,
    "ASSAM": 31205576,
    "BIHAR": 104099452,
    "CHANDIGARH": 1055450,
    "CHHATTISGARH": 25545198,
    "DNHAVELI": 343709,
    "DAMANDDIU": 243247,
    "DELHIUT": 16787941,
    "GOA": 1458545,
    "GUJARAT": 60439692,
    "HARYANA": 25351462,
    "HIMACHALPRADESH": 6864602,
    "JAMMUKASHMIR": 12267032,
    "JHARKHAND": 32988134,
    "KARNATAKA": 61095297,
    "KERALA": 33406061,
    "LAKSHADWEEP": 64473,
    "MADHYAPRADESH": 72626809,
    "MAHARASHTRA": 112374333,
    "MANIPUR": 2855794,
    "MEGHALAYA": 2966889,
    "MIZORAM": 1097206,
    "NAGALAND": 1978502,
    "ODISHA": 41974218,
    "PUDUCHERRY": 1247953,
    "PUNJAB": 27743338,
    "RAJASTHAN": 68548437,
    "SIKKIM": 610577,
    "TAMILNADU": 72147030,
    "TRIPURA": 3673917,
    "UTTARPRADESH": 199812341,
    "UTTARAKHAND": 10086292,
    "WESTBENGAL": 91276115,
}


def _normalize(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper().strip())


def _build_alias_lookup() -> Dict[str, str]:
    lookup = {}
    for standard, aliases in DEFAULT_ALIASES.items():
        for alias in aliases:
            lookup[_normalize(alias)] = standard
    return lookup


def _infer_column_map(columns: List[str], user_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if user_map:
        missing = [std for std, src in user_map.items() if src not in columns]
        if missing:
            raise ValueError(
                "Column map references missing columns: "
                + ", ".join([f"{std}->{user_map[std]}" for std in missing])
            )
        return {std: user_map[std] for std in DEFAULT_ALIASES if std in user_map}

    alias_lookup = _build_alias_lookup()
    normalized_to_original = {}
    for col in columns:
        normalized_to_original.setdefault(_normalize(col), col)

    column_map = {}
    for normalized, original in normalized_to_original.items():
        if normalized in alias_lookup and alias_lookup[normalized] not in column_map:
            column_map[alias_lookup[normalized]] = original

    return column_map


def _load_column_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_total_crimes(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    candidates = [c for c in numeric_cols if c not in {"year", "population"}]
    if not candidates:
        raise ValueError("Unable to compute total_crimes: no numeric crime columns found.")
    df["total_crimes"] = df[candidates].sum(axis=1)
    return df


def _compute_crime_rate(df: pd.DataFrame) -> pd.DataFrame:
    valid_population = df["population"].notna() & (df["population"] > 0)
    df["crime_rate"] = pd.NA
    df.loc[valid_population, "crime_rate"] = (
        (df.loc[valid_population, "total_crimes"] / df.loc[valid_population, "population"]) * 100000.0
    )
    df["crime_rate"] = pd.to_numeric(df["crime_rate"], errors="coerce")
    return df


def _population_from_state_map(state_series: pd.Series) -> pd.Series:
    return state_series.map(lambda s: STATE_POPULATION_2011.get(_normalize(s)))


def _aggregate_population(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["state", "year"])
    pop_sum = grouped["population"].sum(min_count=1)
    pop_max = grouped["population"].max()
    pop_nunique = grouped["population"].nunique(dropna=True)
    pop_count = grouped["population"].count()

    population = pop_sum.copy()
    repeated_total_mask = (pop_nunique == 1) & (pop_count > 1)
    population.loc[repeated_total_mask] = pop_max.loc[repeated_total_mask]

    return population.reset_index(name="population")


def _validate_schema(df: pd.DataFrame, strict: bool = True) -> None:
    errors = []
    for col in RAW_REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        message = "Schema validation failed:\n  - " + "\n  - ".join(errors)
        if strict:
            raise ValueError(message)
        print(message)

    if "year" in df.columns:
        if df["year"].isna().any():
            errors.append("Null values found in year.")
        if not pd.api.types.is_numeric_dtype(df["year"]):
            errors.append("Year column is not numeric.")

    if "total_crimes" in df.columns:
        if df["total_crimes"].isna().any():
            errors.append("Null values found in total_crimes.")

    if errors and strict:
        message = "Schema validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(message)


def run_pipeline(
    input_path: str = "data/raw/dstrIPC_2013.csv",
    output_path: str = "data/processed/clean_crime_data.csv",
    column_map_path: Optional[str] = None,
    compute_total: bool = True,
    strict: bool = True,
    drop_aggregates: bool = True,
) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    user_map = _load_column_map(column_map_path)
    column_map = _infer_column_map(list(df.columns), user_map)

    if "state" in column_map:
        df = df.rename(columns={column_map["state"]: "state"})
    if "district" in column_map:
        df = df.rename(columns={column_map["district"]: "district"})
    if "year" in column_map:
        df = df.rename(columns={column_map["year"]: "year"})
    if "total_crimes" in column_map:
        df = df.rename(columns={column_map["total_crimes"]: "total_crimes"})
    if "population" in column_map:
        df = df.rename(columns={column_map["population"]: "population"})

    if "total_crimes" not in df.columns and compute_total:
        df = _compute_total_crimes(df)

    _validate_schema(df, strict=strict)

    # Keep only columns needed for state-year output.
    keep_cols = [c for c in ["state", "year", "total_crimes", "district", "population"] if c in df.columns]
    df = df[keep_cols].copy()

    # Basic cleaning and normalization.
    df["state"] = df["state"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["total_crimes"] = pd.to_numeric(df["total_crimes"], errors="coerce")
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
    else:
        df["population"] = pd.NA

    df = df.dropna(subset=["state", "year", "total_crimes"])
    df["year"] = df["year"].astype(int)
    df["total_crimes"] = df["total_crimes"].astype(float)

    if drop_aggregates and "district" in df.columns:
        df = df[~df["district"].str.contains(AGGREGATE_DISTRICT_PATTERN, case=False, na=False)]

    # Aggregate to state-year to avoid district-level noise in rate computation.
    aggregated = (
        df.groupby(["state", "year"], as_index=False)
        .agg(
            total_crimes=("total_crimes", "sum"),
        )
        .sort_values(["state", "year"])
        .reset_index(drop=True)
    )

    population_df = _aggregate_population(df)
    aggregated = aggregated.merge(population_df, on=["state", "year"], how="left")
    aggregated["population"] = pd.to_numeric(aggregated["population"], errors="coerce")

    if aggregated["population"].isna().any():
        population_fallback = pd.to_numeric(_population_from_state_map(aggregated["state"]), errors="coerce")
        aggregated["population"] = aggregated["population"].combine_first(population_fallback)

    aggregated.loc[aggregated["population"] <= 0, "population"] = pd.NA
    aggregated = _compute_crime_rate(aggregated)

    stats = aggregated["crime_rate"].dropna()
    if stats.empty:
        raise ValueError("crime_rate could not be computed: all population values are missing or zero.")

    if stats.nunique() <= 1:
        raise ValueError(
            "crime_rate is constant after preprocessing. Check population mapping and source data quality."
        )

    aggregated = aggregated[FINAL_COLUMNS]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aggregated.to_csv(output_path, index=False)

    print("✅ Data pipeline complete.")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Rows:   {len(aggregated)}")
    print(f"   Columns: {', '.join(aggregated.columns)}")
    print("   Crime rate summary (per 100k):")
    print(f"     min:  {stats.min():.4f}")
    print(f"     max:  {stats.max():.4f}")
    print(f"     mean: {stats.mean():.4f}")

    return aggregated


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Crime data pipeline with schema validation.")
    parser.add_argument("--input", default="data/raw/dstrIPC_2013.csv", help="Path to raw CSV.")
    parser.add_argument("--output", default="data/processed/clean_crime_data.csv", help="Path to processed CSV.")
    parser.add_argument("--column-map", default=None, help="Path to JSON file mapping standard -> source columns.")
    parser.add_argument("--no-compute-total", action="store_true", help="Disable computing total_crimes.")
    parser.add_argument("--non-strict", action="store_true", help="Log schema issues instead of raising errors.")
    parser.add_argument("--keep-aggregates", action="store_true", help="Keep aggregate district rows.")

    args = parser.parse_args(argv)
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        column_map_path=args.column_map,
        compute_total=not args.no_compute_total,
        strict=not args.non_strict,
        drop_aggregates=not args.keep_aggregates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
