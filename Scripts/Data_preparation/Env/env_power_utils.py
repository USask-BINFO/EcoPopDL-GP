# env_power_utils.py
# Utilities for fetching NASA POWER daily/hourly data, computing derived variables (VPD, GDD),
# and transforming daily series into fixed-length windowed features for EcoPopDL-GP.
#
# NOTE: This module performs web requests to NASA POWER and (optionally) Nominatim geocoding.
# Run it on a machine with internet access. The functions are pure-Python and have no GPU deps.
#
# Typical usage:
#   from env_power_utils import fetch_power_enriched, windowize_daily, flatten_env
#   df = fetch_power_enriched(lat, lon, "20141001", "20150331", include_par=True, include_sw=True,
#                             include_wind=True, include_dew_or_rh=True, include_precip=True,
#                             compute_vpd=True, compute_gdd=True)
#   win = windowize_daily(df, start="2014-10-01", end="2015-03-31", L=32, align="calendar")
#   X, meta = flatten_env(win, var_cols=["tmax_C","tmin_C","precip_mm","par_allsky","srad_allsky","wind_m_s","vpd_kPa","gdd"])
#
# CLI example:
#   python env_power_utils.py --place "Durgapura, India" --start 20141001 --end 20150331 --out Durgapura_daily.csv --vpdmax
#


import os
import json
import math
import time
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests

# Optional: geocoding (only if you pass a place name instead of lat/lon)
try:
    from geopy.geocoders import Nominatim
    _HAS_GEOPY = True
except Exception:
    _HAS_GEOPY = False


# -------------------------------
# Physics helpers
# -------------------------------
def _saturation_vapor_pressure_kPa(T_C: np.ndarray) -> np.ndarray:
    """FAO-56 Tetens formula for saturation vapor pressure (kPa)."""
    T = np.array(T_C, dtype=float)
    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))


def vpd_from_t_tdew(T_C: np.ndarray, Tdew_C: np.ndarray) -> np.ndarray:
    """VPD (kPa) from mean air temperature ( degC) and dewpoint ( degC)."""
    es = _saturation_vapor_pressure_kPa(T_C)
    ea = _saturation_vapor_pressure_kPa(Tdew_C)
    vpd = es - ea
    return np.clip(vpd, 0, None)


def vpd_from_t_rh(T_C: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
    """VPD (kPa) from mean air temperature ( degC) and relative humidity (%)."""
    es = _saturation_vapor_pressure_kPa(T_C)
    ea = es * (np.array(RH_pct, dtype=float) / 100.0)
    vpd = es - ea
    return np.clip(vpd, 0, None)


def gdd_daily(Tmax_C, Tmin_C, base: float = 10.0, upper: float = 30.0) -> np.ndarray:
    """Growing Degree Days with simple mean and upper cap."""
    Tmax = np.array(Tmax_C, dtype=float)
    Tmin = np.array(Tmin_C, dtype=float)
    tmean = (Tmax + Tmin) / 2.0
    tmean = np.clip(tmean, base, upper)
    return np.clip(tmean - base, 0, None)


# -------------------------------
# NASA POWER fetching
# -------------------------------
DEFAULT_PARAMS_DAILY = [
    "T2M", "T2M_MAX", "T2M_MIN",
    "T2MDEW", "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_PAR_TOT"
]

# -------------------------------
# Simple astronomy (FAO-56 + NOAA)
# -------------------------------
_GSC_MJ_m2_min = 0.0820  # solar constant (MJ m^-2 min^-1)

def _declination_rad(J: int) -> float:
    return 0.409 * math.sin((2.0 * math.pi / 365.0) * J - 1.39)

def _dr(J: int) -> float:
    return 1.0 + 0.033 * math.cos((2.0 * math.pi / 365.0) * J)

def _sunset_hour_angle_rad(phi_rad: float, delta: float) -> float:
    x = -math.tan(phi_rad) * math.tan(delta)
    return math.acos(max(min(x, 1.0), -1.0))

def _toa_Ra_MJ_m2_day(phi_rad: float, J: int) -> float:
    delta = _declination_rad(J); dr = _dr(J)
    omega_s = _sunset_hour_angle_rad(phi_rad, delta)
    Ra = (24.0 * 60.0 / math.pi) * _GSC_MJ_m2_min * dr * (
        omega_s * math.sin(phi_rad) * math.sin(delta) +
        math.cos(phi_rad) * math.cos(delta) * math.sin(omega_s)
    )
    return max(Ra, 1e-9)  # avoid divide-by-zero

def _daylength_hours_noaa(phi_rad: float, J: int, elev_m: float = 0.0) -> float:
    # NOAA sunrise/sunset with -0.833 deg apparent solar elevation
    delta = _declination_rad(J)
    alt = -0.833 - 2.076 * math.sqrt(max(elev_m, 0.0)) / 60.0
    sin_zen = math.sin(math.radians(alt))
    cos_omega0 = (sin_zen - math.sin(phi_rad) * math.sin(delta)) / (math.cos(phi_rad) * math.cos(delta))
    cos_omega0 = max(min(cos_omega0, 1.0), -1.0)
    omega0 = math.acos(cos_omega0)
    return 2.0 * math.degrees(omega0) / 15.0


def _nasa_power_daily_point(lat: float, lon: float, start: str, end: str,
                            params: List[str],
                            community: str = "AG",
                            time_standard: str = "UTC",
                            timeout: int = 60) -> pd.DataFrame:
    """Fetch NASA POWER daily variables for a lat/lon and return DataFrame with 'date' column."""
    param_str = ",".join(params)
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={param_str}"
        f"&community={community}"
        f"&latitude={lat}&longitude={lon}"
        f"&start={start}&end={end}"
        f"&time-standard={time_standard}"
        "&format=JSON"
    )
    headers = {"User-Agent": "EcoPopDL-GP/1.0 (contact: your_email@example.com)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "properties" not in data or "parameter" not in data["properties"]:
        raise ValueError(f"POWER response missing expected fields: keys={list(data.keys())}")
    p = data["properties"]["parameter"]
    # date index
    first_param = next(iter(p))
    dates = sorted(p[first_param].keys())
    df = pd.DataFrame({"date": pd.to_datetime(dates, format="%Y%m%d")})
    # attach
    for prm in params:
        series = p.get(prm, None)
        if series is None:
            df[prm] = np.nan
        else:
            df[prm] = df["date"].dt.strftime("%Y%m%d").map(series).astype(float)
    return df.sort_values("date").reset_index(drop=True)


def _nasa_power_hourly_point(lat: float, lon: float, start: str, end: str,
                             params: List[str],
                             community: str = "AG",
                             time_standard: str = "LST",
                             timeout: int = 90) -> pd.DataFrame:
    """Fetch NASA POWER hourly variables for a lat/lon. Returns DataFrame with 'dt' (timestamp)."""
    param_str = ",".join(params)
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters={param_str}"
        f"&community={community}"
        f"&latitude={lat}&longitude={lon}"
        f"&start={start}&end={end}"
        f"&time-standard={time_standard}"
        "&format=JSON"
    )
    headers = {"User-Agent": "EcoPopDL-GP/1.0 (contact: your_email@example.com)"
               }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    p = data["properties"]["parameter"]
    # build hourly df
    idx_keys = sorted(next(iter(p.values())).keys())  # 'YYYYMMDDHH'
    hh = pd.DataFrame({"dt": pd.to_datetime(idx_keys, format="%Y%m%d%H")})
    for prm in params:
        series = p.get(prm, None)
        if series is None:
            hh[prm] = np.nan
        else:
            hh[prm] = hh["dt"].dt.strftime("%Y%m%d%H").map(series).astype(float)
    return hh.sort_values("dt").reset_index(drop=True)

def fetch_power_enriched(lat, lon, start, end,
                         include_par=False, include_sw=False, include_wind=False,
                         include_dew_or_rh=False, include_precip=False,
                         compute_vpd=False, compute_gdd=False,
                         hourly_vpdmax=False,
                         include_cloud=False, include_daylen=False):
    """
    Fetch daily POWER variables and add derived columns:
    - vpd_kPa (from T2M + T2MDEW preferred, else RH2M)
    - gdd, gdd_cum
    - optional vpdmax_kPa from hourly T2M/RH2M

    Parameters:
      start, end: 'YYYYMMDD'
    """
    params = ["T2M", "T2M_MAX", "T2M_MIN"]
    if include_dew_or_rh:
        params += ["T2MDEW", "RH2M"]
    if include_wind:
        params += ["WS2M"]
    if include_precip:
        params += ["PRECTOTCORR"]
    if include_sw:
        params += ["ALLSKY_SFC_SW_DWN"]
    if include_par:
        params += ["ALLSKY_SFC_PAR_TOT"]
    
    # We'll compute kt = ALLSKY_SFC_SW_DWN / ALLSKY_TOA_SW_DWN
    if include_sw:
        params.append("ALLSKY_TOA_SW_DWN")


    wind_col = None
    try:
        df = _nasa_power_daily_point(lat, lon, start, end, params,
                                     community="AG", time_standard="UTC")
        if include_wind:
            wind_col = "WS2M" if "WS2M" in params else ("WS10M" if "WS10M" in params else None)
    except requests.HTTPError:
        # If WS2M is not allowed for some historical products, try WS10M
        if include_wind and "WS2M" in params:
            params = [("WS10M" if x == "WS2M" else x) for x in params]
            df = _nasa_power_daily_point(lat, lon, start, end, params,
                                         community="AG", time_standard="UTC")
            wind_col = "WS10M"
        else:
            raise

    # rename
    rename_map = {
        "T2M": "tmean_C",
        "T2M_MAX": "tmax_C",
        "T2M_MIN": "tmin_C",
        "T2MDEW": "tdew_C",
        "RH2M": "rh_pct",
        "PRECTOTCORR": "precip_mm",
        "ALLSKY_SFC_SW_DWN": "srad_allsky",
        "ALLSKY_SFC_PAR_TOT": "par_allsky",
        "ALLSKY_TOA_SW_DWN": "srad_toa",
        "CLD_AMT": "cloud_amt_raw",
        "SG_DAY_HOUR_AVG": "daylength_h"
    }
    if wind_col:
        rename_map[wind_col] = "wind_m_s"
    df = df.rename(columns=rename_map)

    # derived
    if compute_vpd:
        if "tdew_C" in df.columns and df["tdew_C"].notnull().any():
            df["vpd_kPa"] = vpd_from_t_tdew(df["tmean_C"].values, df["tdew_C"].values)
        elif "rh_pct" in df.columns and df["rh_pct"].notnull().any():
            df["vpd_kPa"] = vpd_from_t_rh(df["tmean_C"].values, df["rh_pct"].values)
        else:
            df["vpd_kPa"] = np.nan

    if compute_gdd:
        df["gdd"] = gdd_daily(df["tmax_C"].values, df["tmin_C"].values, base=10.0, upper=30.0)
        df["gdd_cum"] = np.cumsum(df["gdd"].values)

    if hourly_vpdmax:
        try:
            hh = _nasa_power_hourly_point(lat, lon, start, end, params=["T2M", "RH2M"], community="AG", time_standard="LST")
            hh["VPD_kPa"] = vpd_from_t_rh(hh["T2M"].values, hh["RH2M"].values)
            daily_vpdmax = (hh.assign(date=hh["dt"].dt.date)
                              .groupby("date", as_index=False)["VPD_kPa"].max()
                              .rename(columns={"VPD_kPa": "vpdmax_kPa"}))
            daily_vpdmax["date"] = pd.to_datetime(daily_vpdmax["date"])
            df = df.merge(daily_vpdmax, on="date", how="left")
        except Exception as e:
            # If hourly fails, just proceed without vpdmax
            pass
    # ---- New: clearness index (kt), cloudiness proxy, and day length ----
    phi_rad = math.radians(float(lat))
    J = df["date"].dt.dayofyear.to_numpy(dtype=int)

    # Get a TOA denominator: prefer ALLSKY_TOA_SW_DWN if you kept it; else FAO-56 Ra
    if "srad_allsky" in df.columns:
        if "ALLSKY_TOA_SW_DWN" in df.columns:
            denom = df["ALLSKY_TOA_SW_DWN"].astype(float).to_numpy()
        else:
            denom = np.array([_toa_Ra_MJ_m2_day(phi_rad, int(j)) for j in J])
        with np.errstate(divide="ignore", invalid="ignore"):
            df["kt"] = np.clip(df["srad_allsky"].astype(float) / denom, 0.0, 1.2)  # clearness index (POWER definition)

    # Cloudiness proxy (percent): 100 x (1 - kt), clipped
    if "kt" in df.columns and include_cloud:
        df["cloud_pct"] = np.clip((1.0 - df["kt"]) * 100.0, 0.0, 100.0)

    # Astronomical day length (hours)
    if include_daylen:
        df["daylength_h"] = np.array([_daylength_hours_noaa(phi_rad, int(j)) for j in J])

        
    return df.sort_values("date").reset_index(drop=True)


# -------------------------------
# Windowing / Flattening
# -------------------------------
def windowize_daily(df_daily: pd.DataFrame,
                    start: Optional[str] = None,
                    end: Optional[str] = None,
                    L: Optional[int] = None,
                    align: str = "calendar",
                    gdd_edges: Optional[List[float]] = None,
                    var_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate daily variables into L windows (calendar or thermal).
    Returns a DataFrame with columns: ['win', <var_cols...>] where 'win' is 0..L-1 (or len(edges)-1).

    align:
      - "calendar": split [start, end] into L equal-length day windows.
      - "thermal": split by cumulative GDD bins; if gdd_edges is None and L is given,
                   uses quantile bins of gdd_cum into L equal-count bins.
    """
    df = df_daily.copy()
    if start is None:
        start = df["date"].min().strftime("%Y-%m-%d")
    if end is None:
        end = df["date"].max().strftime("%Y-%m-%d")
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    if df.empty:
        raise ValueError("No daily rows in the requested [start, end] interval.")

    if var_cols is None:
        # default to all numeric columns except 'date' and obvious cumulative columns
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        var_cols = [c for c in numeric if c not in ("gdd_cum",)]

    if align == "calendar":
        if L is None:
            raise ValueError("For 'calendar' alignment, please provide L (number of windows).")
        total_days = (end_dt - start_dt).days + 1
        # Build equal day windows
        edges = [start_dt + pd.Timedelta(days=math.floor(k * total_days / L)) for k in range(L)]
        edges.append(end_dt + pd.Timedelta(days=1))  # closed on left, open on right bins
        df["win"] = pd.cut(df["date"], bins=edges, right=False, labels=False)
    elif align == "thermal":
        if "gdd_cum" not in df.columns:
            raise ValueError("Thermal alignment requires 'gdd_cum' in the daily table. Re-run with compute_gdd=True.")
        if gdd_edges is None:
            if L is None:
                raise ValueError("Provide L (number of windows) or gdd_edges for thermal alignment.")
            # quantile-based edges on gdd_cum
            quantiles = np.linspace(0, 1, L + 1)
            gdd_edges = np.quantile(df["gdd_cum"].values, quantiles).tolist()
        df["win"] = pd.cut(df["gdd_cum"], bins=gdd_edges, right=True, labels=False, include_lowest=True)
    else:
        raise ValueError("align must be 'calendar' or 'thermal'.")

    agg = df.groupby("win", as_index=False)[var_cols].mean()
    # ensure full 0..L-1 rows exist
    max_win = (L - 1) if L is not None else (len(gdd_edges) - 2)
    full = pd.DataFrame({"win": np.arange(max_win + 1, dtype=int)})
    agg = full.merge(agg, on="win", how="left").sort_values("win").reset_index(drop=True)
    return agg


def flatten_env(win_df: pd.DataFrame,
                var_cols: List[str],
                prefix: str = "E",
                order: str = "time-major") -> Tuple[pd.DataFrame, Dict]:
    """
    Flatten windowed env DataFrame into a single row of concatenated features.
    order = 'time-major' -> [win0 all vars, win1 all vars, ...]
          = 'var-major'  -> [var1 all wins, var2 all wins, ...]
    Returns (X_df, meta_dict) where X_df has columns like f"{prefix}_{var}_{k:02d}".
    """
    assert "win" in win_df.columns, "win_df must have a 'win' column from windowize_daily()."
    L = int(win_df["win"].max()) + 1 if len(win_df) else 0
    cols = []
    values = []
    if order == "time-major":
        for k in range(L):
            row = win_df.loc[win_df["win"] == k]
            if row.empty:
                row = pd.Series({v: np.nan for v in var_cols})
            else:
                row = row[var_cols].iloc[0]
            for v in var_cols:
                cols.append(f"{prefix}_{v}_{k:02d}")
                values.append(float(row.get(v, np.nan)))
    elif order == "var-major":
        for v in var_cols:
            series = win_df.set_index("win")[v].reindex(range(L))
            for k in range(L):
                cols.append(f"{prefix}_{v}_{k:02d}")
                values.append(float(series.iloc[k]))
    else:
        raise ValueError("order must be 'time-major' or 'var-major'.")

    X = pd.DataFrame([values], columns=cols)
    meta = {"L": L, "vars_per_window": len(var_cols), "var_cols": var_cols, "order": order, "prefix": prefix}
    return X, meta


def geocode_place(place: str) -> Tuple[float, float]:
    """Geocode a place name using Nominatim."""
    if not _HAS_GEOPY:
        raise ImportError("geopy is not installed. Please `pip install geopy`.")
    geolocator = Nominatim(user_agent="EcoPopDL-GP_geocoder")
    loc = geolocator.geocode(place)
    if loc is None:
        raise ValueError(f"Could not geocode: {place}")
    return float(loc.latitude), float(loc.longitude)


# -------------------------------
# CLI
# -------------------------------
def _main():
    ap = argparse.ArgumentParser(description="Fetch NASA POWER daily data with derived variables and save CSV.")
    ap.add_argument("--place", type=str, default=None, help="Place name to geocode (alternative to lat/lon).")
    ap.add_argument("--lat", type=float, default=None, help="Latitude (if not using --place).")
    ap.add_argument("--lon", type=float, default=None, help="Longitude (if not using --place).")
    ap.add_argument("--start", type=str, required=True, help="Start date YYYYMMDD.")
    ap.add_argument("--end", type=str, required=True, help="End date YYYYMMDD.")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path.")
    ap.add_argument("--vpdmax", action="store_true", help="Include daily VPDmax derived from hourly T/RH.")
    ap.add_argument("--no-par", dest="par", action="store_false", help="Omit PAR (ALLSKY_SFC_PAR_TOT).")
    ap.add_argument("--no-sw", dest="sw", action="store_false", help="Omit shortwave (ALLSKY_SFC_SW_DWN).")
    ap.add_argument("--no-wind", dest="wind", action="store_false", help="Omit wind speed.")
    ap.add_argument("--no-precip", dest="precip", action="store_false", help="Omit precipitation.")
    ap.add_argument("--no-dew-rh", dest="dew_or_rh", action="store_false", help="Omit dewpoint/RH (disables VPD).")
    ap.set_defaults(par=True, sw=True, wind=True, precip=True, dew_or_rh=True)

    args = ap.parse_args()

    if args.place:
        lat, lon = geocode_place(args.place)
    else:
        if args.lat is None or args.lon is None:
            raise SystemExit("Provide either --place or both --lat and --lon.")
        lat, lon = args.lat, args.lon

    df = fetch_power_enriched(
        lat, lon, args.start, args.end,
        include_par=args.par, include_sw=args.sw, include_wind=args.wind,
        include_dew_or_rh=args.dew_or_rh, include_precip=args.precip,
        compute_vpd=True, compute_gdd=True, hourly_vpdmax=args.vpdmax
    )
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    _main()
