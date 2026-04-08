
# build_env_matrix_v2.py
# Same functionality as build_env_matrix.py, but adds:
# - Progress bars (tqdm, with a graceful fallback to prints)
# - On-disk caching of NASA POWER daily fetches (to avoid re-downloading)
# - Optional parallel downloads with --workers (be considerate to APIs)
# - Geocoding cache to avoid repeated lookups
#
# Usage example:
# python3 build_env_matrix_v2.py   --trials D3/Trial_data_D3_SW.csv   --L 32 --align calendar   --out-matrix d2_env_matrix.csv --out-meta env_meta.json   --cache-dir .env_cache --geocode-cache .geocode_cache.json   --workers 4   --vars tmax_C tmin_C precip_mm par_allsky srad_allsky wind_m_s vpd_kPa gdd cloud_pct kt daylength_h

#

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Tuple

import pandas as pd

from env_power_utils import (
    geocode_place, fetch_power_enriched, windowize_daily, flatten_env
)

# --------- tqdm (optional) ----------
try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False
    def tqdm(x, **kwargs):
        # fallback: identity iterator
        return x


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(obj: Dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _norm_vars(vars_list: List[str]) -> List[str]:
    return sorted([str(v).strip() for v in vars_list])


def _site_cache_key(lat: float, lon: float, start: str, end: str,
                    vars_list: List[str], vpdmax: bool) -> str:
    v = "-".join(_norm_vars(vars_list))
    key_str = f"{round(float(lat), 4)},{round(float(lon), 4)}|{start}|{end}|{v}|vpdmax={int(bool(vpdmax))}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_from_cache(cache_dir: str, key: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, key + ".csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return None


def _save_to_cache(cache_dir: str, key: str, df: pd.DataFrame):
    path = os.path.join(cache_dir, key + ".csv")
    df.to_csv(path, index=False)


def _infer_latlon(row, geocode_cache: Dict) -> Tuple[float, float]:
    if pd.notnull(row.get("lat")) and pd.notnull(row.get("lon")):
        return float(row["lat"]), float(row["lon"])

    # accept several possible column names for the site label
    place = None
    for k in ("place", "location", "Location"):
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            place = v.strip()
            break

    if not place:
        raise ValueError(f"Row {row.get('sample_id')} missing lat/lon and place/location.")

    # geocode cache first
    if place in geocode_cache:
        lat, lon = geocode_cache[place]
        return float(lat), float(lon)

    # geocode live
    lat, lon = geocode_place(place)
    geocode_cache[place] = [lat, lon]
    return float(lat), float(lon)



def main():
    ap = argparse.ArgumentParser(description="Build windowed environment matrix from NASA POWER daily data (with caching and progress).")
    ap.add_argument(
        "--trials",
        required=True,
        help=(
            "CSV with columns: sample_id, start, end, and either lat/lon or "
            "place/location/Location for geocoding."
        ),
    )
    ap.add_argument("--L", type=int, required=True, help="Number of windows per sample (e.g., 32).")
    ap.add_argument("--align", type=str, default="calendar", choices=["calendar","thermal"], help="Windowing mode.")
    ap.add_argument("--out-matrix", required=True, help="Output CSV path for flattened env matrix.")
    ap.add_argument("--out-meta", required=True, help="Output JSON path for env meta (L, vars_per_window, var_cols).")
    ap.add_argument("--vpdmax", action="store_true", help="Include daily VPDmax via hourly (slower).")
    ap.add_argument("--vars", nargs="+", default=["tmax_C","tmin_C","precip_mm","par_allsky","srad_allsky","wind_m_s","vpd_kPa","gdd","cloud_pct","kt","daylength_h"
    ],help="Variables to include per window in this order.")
    ap.add_argument("--cache-dir", default=".env_cache", help="Directory for on-disk cache (POWER daily responses).")
    ap.add_argument("--geocode-cache", default=".geocode_cache.json", help="JSON file to cache place->(lat,lon).")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers for downloads (be considerate to APIs).")
    ap.add_argument("--retry", type=int, default=3, help="Retry attempts per sample on transient failures.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between attempts (politeness).")

    args = ap.parse_args()

    _ensure_dir(args.cache_dir)
    geocode_cache = _load_json(args.geocode_cache)

    trials = pd.read_csv(args.trials)
    # Normalize and prepare rows
    rows = []
    for i, row in trials.iterrows():
        sample_id = row["sample_id"]
        lat, lon = _infer_latlon(row, geocode_cache)

        # Friendly location label for output CSV
        location = None
        for k in ("location", "Location", "place"):
            if k in trials.columns:
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    location = v.strip()
                    break

        # Normalize/compute dates and Year
        start_ts  = pd.to_datetime(row["start"])
        end_ts    = pd.to_datetime(row["end"])
        start_iso = start_ts.strftime("%Y-%m-%d")
        end_iso   = end_ts.strftime("%Y-%m-%d")
        start     = start_iso.replace("-", "")
        end       = end_iso.replace("-", "")
        year      = int(start_ts.year)  # <- Year from start

        site_key = _site_cache_key(lat, lon, start, end, args.vars, args.vpdmax)
        rows.append({
            "sample_id": sample_id,
            "lat": lat, "lon": lon,
            "start": start, "end": end,
            "start_iso": start_iso, "end_iso": end_iso,
            "location": location,          # NEW
            "year": year,                  # NEW
            "site_key": site_key
        })


    # Persist geocode cache early
    _save_json(geocode_cache, args.geocode_cache)

    # Deduplicate sites to minimize downloads
    unique_sites = {}
    for r in rows:
        unique_sites.setdefault(r["site_key"], {"lat": r["lat"], "lon": r["lon"], "start": r["start"], "end": r["end"]})

    # Step 1: fetch (or load) daily data per unique site
    site_data: Dict[str, pd.DataFrame] = {}

    def _fetch_site(key: str, lat: float, lon: float, start: str, end: str) -> Tuple[str, pd.DataFrame]:
        # cache first
        df_cached = _load_from_cache(args.cache_dir, key)
        if df_cached is not None:
            return key, df_cached
        # fetch with retries
        last_err = None
        for attempt in range(1, args.retry + 1):
            try:
                df = fetch_power_enriched(
                    lat, lon, start, end,
                    include_par=("par_allsky" in args.vars),
                    include_sw=("srad_allsky" in args.vars),
                    include_wind=("wind_m_s" in args.vars),
                    include_dew_or_rh=("vpd_kPa" in args.vars or "vpdmax_kPa" in args.vars),
                    include_precip=("precip_mm" in args.vars),
                    compute_vpd=("vpd_kPa" in args.vars),
                    compute_gdd=("gdd" in args.vars),
                    hourly_vpdmax=args.vpdmax,
                    include_cloud=("cloud_pct" in args.vars or "kt" in args.vars),
                    include_daylen=("daylength_h" in args.vars)
                )

                _save_to_cache(args.cache_dir, key, df)
                return key, df
            except Exception as e:
                last_err = e
                time.sleep(args.sleep * (2 ** (attempt - 1)))  # simple backoff
        # if we got here, raise last error
        raise last_err

    # Progress: downloads (possibly parallel)
    keys = list(unique_sites.keys())
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_fetch_site, k, unique_sites[k]["lat"], unique_sites[k]["lon"],
                                 unique_sites[k]["start"], unique_sites[k]["end"]): k for k in keys}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching POWER daily", ncols=100):
                k = futures[fut]
                key, df = fut.result()
                site_data[key] = df
    else:
        for k in tqdm(keys, desc="Fetching POWER daily", ncols=100):
            key, df = _fetch_site(k, unique_sites[k]["lat"], unique_sites[k]["lon"],
                                  unique_sites[k]["start"], unique_sites[k]["end"])
            site_data[key] = df

    # Step 2: windowize + flatten for all rows
    out_rows = []
    first_meta = None
    for r in tqdm(rows, desc="Windowize & flatten", ncols=100):
        df = site_data[r["site_key"]]
        win = windowize_daily(df,
                              start=r["start_iso"],
                              end=r["end_iso"],
                              L=args.L,
                              align=args.align,
                              var_cols=args.vars)
        X, meta = flatten_env(win, var_cols=args.vars, prefix="E", order="time-major")
        X.insert(0, "sample_id", r["sample_id"])
        X.insert(1, "Location", r.get("location") or "")  # NEW
        X.insert(2, "Year", int(r["year"]))               # NEW
        out_rows.append(X)

        if first_meta is None:
            first_meta = meta

    matrix = pd.concat(out_rows, axis=0).reset_index(drop=True)
    matrix.to_csv(args.out_matrix, index=False)
    with open(args.out_meta, "w") as f:
        json.dump(first_meta, f, indent=2)

    print(f"Wrote matrix to {args.out_matrix} with shape {matrix.shape}")
    print(f"Wrote meta to {args.out_meta}: {first_meta}")


if __name__ == "__main__":
    main()
