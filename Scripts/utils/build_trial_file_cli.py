#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def require_column(df: pd.DataFrame, name: str, flag: str) -> None:
    if name and name not in df.columns:
        raise ValueError(f"{flag} column '{name}' not found in input file.")


def parse_mm_dd(value: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(value, "%m-%d")
    except ValueError as exc:
        raise ValueError(f"Expected MM-DD value, got '{value}'.") from exc
    return parsed.month, parsed.day


def build_dates(
    df: pd.DataFrame,
    year_col: str,
    start_col: str,
    end_col: str,
    season_start_mm_dd: str,
    season_end_mm_dd: str,
    season_length_days: int | None,
) -> tuple[pd.Series, pd.Series]:
    if start_col and end_col:
        start = pd.to_datetime(df[start_col], errors="raise")
        end = pd.to_datetime(df[end_col], errors="raise")
        return start, end

    if not year_col:
        raise ValueError(
            "Provide --year-col when building start/end dates from season settings."
        )
    if not season_start_mm_dd:
        raise ValueError(
            "Provide --season-start-mm-dd when no explicit --start-col/--end-col are supplied."
        )
    if not season_end_mm_dd and season_length_days is None:
        raise ValueError(
            "Provide either --season-end-mm-dd or --season-length-days when no explicit start/end columns are supplied."
        )

    start_month, start_day = parse_mm_dd(season_start_mm_dd)
    end_month = end_day = None
    if season_end_mm_dd:
        end_month, end_day = parse_mm_dd(season_end_mm_dd)

    years = pd.to_numeric(df[year_col], errors="raise").astype(int)
    start_vals = []
    end_vals = []
    for year in years:
        start_dt = datetime(year, start_month, start_day)
        if season_end_mm_dd:
            end_year = year
            if (end_month, end_day) < (start_month, start_day):
                end_year += 1
            end_dt = datetime(end_year, end_month, end_day)
        else:
            if season_length_days is None or season_length_days < 1:
                raise ValueError("--season-length-days must be a positive integer.")
            end_dt = start_dt + timedelta(days=int(season_length_days) - 1)
        start_vals.append(start_dt)
        end_vals.append(end_dt)

    return pd.Series(start_vals, index=df.index), pd.Series(end_vals, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a trial window CSV for environment extraction from a simpler "
            "phenotype-like table."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV/TSV/TXT table.")
    parser.add_argument("--output", required=True, help="Output trial CSV path.")
    parser.add_argument("--sample-col", required=True, help="Sample ID column.")
    parser.add_argument("--location-col", default="", help="Location/site column.")
    parser.add_argument("--year-col", default="", help="Year column used with season settings.")
    parser.add_argument("--lat-col", default="", help="Latitude column.")
    parser.add_argument("--lon-col", default="", help="Longitude column.")
    parser.add_argument("--place-col", default="", help="Place-name column used for geocoding.")
    parser.add_argument("--start-col", default="", help="Explicit season start-date column.")
    parser.add_argument("--end-col", default="", help="Explicit season end-date column.")
    parser.add_argument(
        "--season-start-mm-dd",
        default="",
        help="Season start in MM-DD form when explicit dates are absent.",
    )
    parser.add_argument(
        "--season-end-mm-dd",
        default="",
        help="Season end in MM-DD form when explicit dates are absent.",
    )
    parser.add_argument(
        "--season-length-days",
        type=int,
        default=None,
        help="Season length when explicit end dates are absent.",
    )
    args = parser.parse_args()

    df = read_table(args.input)
    for col_name, flag in (
        (args.sample_col, "--sample-col"),
        (args.location_col, "--location-col"),
        (args.year_col, "--year-col"),
        (args.lat_col, "--lat-col"),
        (args.lon_col, "--lon-col"),
        (args.place_col, "--place-col"),
        (args.start_col, "--start-col"),
        (args.end_col, "--end-col"),
    ):
        require_column(df, col_name, flag)

    if bool(args.lat_col) != bool(args.lon_col):
        raise ValueError("Provide both --lat-col and --lon-col together.")
    if not args.location_col and not args.place_col and not args.lat_col:
        raise ValueError(
            "Provide at least one of --location-col, --place-col, or --lat-col/--lon-col."
        )

    start, end = build_dates(
        df=df,
        year_col=args.year_col,
        start_col=args.start_col,
        end_col=args.end_col,
        season_start_mm_dd=args.season_start_mm_dd,
        season_end_mm_dd=args.season_end_mm_dd,
        season_length_days=args.season_length_days,
    )

    out = pd.DataFrame({"sample_id": df[args.sample_col].astype(str).str.strip()})
    if args.location_col:
        out["Location"] = df[args.location_col].astype(str).str.strip()
    if args.place_col:
        place = df[args.place_col].astype(str).str.strip()
        if "Location" not in out.columns or not place.equals(out["Location"]):
            out["place"] = place
    if args.lat_col:
        out["lat"] = pd.to_numeric(df[args.lat_col], errors="raise")
        out["lon"] = pd.to_numeric(df[args.lon_col], errors="raise")

    out["start"] = start.dt.strftime("%Y-%m-%d")
    out["end"] = end.dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates().sort_values(["sample_id", "start", "end"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Wrote {len(out)} trial rows to {output_path}")


if __name__ == "__main__":
    main()
