#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JCOPE / JAMSTEC Fortran (GrADS) binary -> NetCDF (sigma as-is)

- Read: get nx, ny, nz, undef, options from .ctl
- Binary: assumes single variable, single time (TT_, EGT_, etc). Supports sequential (record markers).
- Output: NetCDF (dims = time, sigma, y, x) but if nz==1 then (time, y, x)
- σ-coordinate: if ZDEF levels exist, they are used as sigma coordinate (units="1")
- Missing values: UNDEF masked to NaN. Add _FillValue=UNDEF

Example usage (CLI):
  python jcope_bin_to_netcdf.py --ctl TT-2.ctl --bin TT_202102251600.bin -o TT_202102251600.nc
  python jcope_bin_to_netcdf.py --ctl EGT.ctl --bin EGT_202102262300.bin -o EGT_202102262300.nc
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import xarray as xr
except Exception as e:
    raise SystemExit("xarray is required (pip install xarray).") from e


# --------------------------
# Helper: infer time from filename
# --------------------------
def guess_time_from_filename(bin_path: str):
    """
    Guess timestamp from trailing digits in filename:
      - 12 digits: YYYYMMDDHHMM
      - 10 digits: YYYYMMDDHH   (minute = 00)
      -  8 digits: YYYYMMDD     (time = 00:00)
    Returns None if not found.
    """
    name = Path(bin_path).stem
    m = re.search(r'(\d{12})$', name) or re.search(r'(\d{10})$', name) or re.search(r'(\d{8})$', name)
    if not m:
        m = re.search(r'(\d{12})', name) or re.search(r'(\d{10})', name) or re.search(r'(\d{8})', name)
        if not m:
            return None

    s = m.group(1)
    if len(s) == 12:
        dt = datetime.strptime(s, "%Y%m%d%H%M")
    elif len(s) == 10:
        dt = datetime.strptime(s, "%Y%m%d%H")
    else:  # 8
        dt = datetime.strptime(s, "%Y%m%d")
    return np.array([np.datetime64(dt)], dtype="datetime64[ns]")  # length-1 array


# --------------------------
# .CTL parser
# --------------------------
def _split(line: str) -> List[str]:
    return re.split(r"\s+", line.strip())

def parse_ctl(ctl_path: str) -> Dict:
    """
    Parse a GrADS .ctl file and return info dict
    """
    txt = Path(ctl_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    # Remove comments (lines starting with * or inline *...)
    lines = [re.sub(r"\*.*$", "", L).strip() for L in txt]
    lines = [L for L in lines if L]

    info = {
        "dset": None,
        "undef": None,
        "options": [],
        "xdef": None,
        "ydef": None,
        "zdef": None,
        "tdef": None,
        "vars": [],
    }

    it = iter(lines)
    for line in it:
        low = line.lower()
        if low.startswith("dset"):
            m = re.match(r"dset\s+(.+)$", line, flags=re.I)
            if m:
                info["dset"] = m.group(1).strip().strip("'").strip('"')
        elif low.startswith("undef"):
            parts = _split(line)
            if len(parts) >= 2:
                try:
                    info["undef"] = float(parts[1])
                except Exception:
                    pass
        elif low.startswith("options"):
            parts = _split(line)
            info["options"] = [p.lower() for p in parts[1:]]
        elif low.startswith("xdef"):
            info["xdef"] = _parse_def_axis(line, it, key="xdef")
        elif low.startswith("ydef"):
            info["ydef"] = _parse_def_axis(line, it, key="ydef")
        elif low.startswith("zdef"):
            info["zdef"] = _parse_def_axis(line, it, key="zdef")
        elif low.startswith("tdef"):
            info["tdef"] = _parse_tdef(line, it)
        elif low.startswith("vars"):
            while True:
                vline = next(it)
                if vline.lower().startswith("endvars"):
                    break
                toks = _split(vline)
                if len(toks) >= 2:
                    name = toks[0]
                    try:
                        nlev = int(toks[1])
                    except Exception:
                        nlev = 1
                    desc = " ".join(toks[2:]) if len(toks) > 2 else ""
                    info["vars"].append((name, nlev, desc))

    return info


def _parse_def_axis(first_line: str, it, key: str) -> Dict:
    """
    Parse XDEF/YDEF/ZDEF line(s)
    """
    parts = _split(first_line)
    n = int(parts[1])
    typ = parts[2].lower()
    if typ == "linear":
        v0 = float(parts[3]); dv = float(parts[4])
        # calcule en float64 puis cast en float32 pour rester cohérent
        vals = (v0 + dv * np.arange(n, dtype=np.float64))#.astype(np.float32)
        return {"n": n, "type": "linear", "vals": vals}
    elif typ == "levels":
        remain = " ".join(parts[3:])
        while len(re.findall(r"[-+0-9.eE]+", remain)) < n:
            try:
                remain += " " + next(it)
            except StopIteration:
                break
        nums = re.findall(r"[-+0-9.eE]+", remain)
        vals = np.array(list(map(float, nums[:n])), dtype=np.float32)#.astype(np.float32)
        return {"n": n, "type": "levels", "vals": vals}
    else:
        raise ValueError(f"Unsupported {key} type: {typ}")



def _parse_tdef(first_line: str, it) -> Dict:
    """
    Parse TDEF line
    """
    parts = _split(first_line)
    n = int(parts[1])
    typ = parts[2].lower()
    if typ == "linear":
        start = parts[3]
        dt_str = parts[4] if len(parts) > 4 else "1hr"
        return {"n": n, "type": "linear", "start": start, "dt": dt_str}
    elif typ == "levels":
        remain = " ".join(parts[3:])
        toks = re.findall(r"[^\s]+", remain)
        return {"n": n, "type": "levels", "levels": toks[:n]}
    else:
        raise ValueError(f"Unsupported TDEF type: {typ}")


# --------------------------
# Binary reader
# --------------------------
def _dtype_from_options(options: List[str]) -> np.dtype:
    endian = ">" if "big_endian" in options else "<" if "little_endian" in options else ">"
    return np.dtype(endian + "f4")

def read_grads_singlevar_singletime(
    bin_path: str,
    ctl: Dict,
    var_name: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], Dict]:
    """
    Read single variable, single time binary into ndarray.
    Returns:
      data : ndarray, shape = (nz, ny, nx) or (ny, nx)
      dims : list[str]
      coords : dict
      meta : dict
    """
    nx = ctl["xdef"]["n"]; ny = ctl["ydef"]["n"]
    nz = ctl["zdef"]["n"] if ctl.get("zdef") else 1
    nt = ctl["tdef"]["n"] if ctl.get("tdef") else 1

    options = ctl.get("options", [])
    dtype = _dtype_from_options(options)
    undef = ctl.get("undef", 1.0e20)

    x_vals = ctl["xdef"]["vals"]
    y_vals = ctl["ydef"]["vals"]
    sigma_vals = ctl["zdef"]["vals"] if ctl.get("zdef") else None

    a = np.fromfile(bin_path, dtype=dtype)
    expected_vals = nx * ny * nz
    if a.size != expected_vals:
        raise ValueError(
            f"Size mismatch: file has {a.size}, expected {expected_vals}."
        )
    if nz == 1:
        arr = a.reshape((nx, ny), order="F").T
        dims = ["y", "x"]; coords = {"y": y_vals, "x": x_vals}
    else:
        arr = a.reshape((nx, ny, nz), order="F").transpose(2, 1, 0)
        dims = ["sigma", "y", "x"]; coords = {"sigma": sigma_vals, "y": y_vals, "x": x_vals}

    data = np.where(np.abs(arr) >= 0.9 * float(undef), np.nan, arr).astype(np.float32)

    meta = {"undef": float(undef), "options": options, "nx": nx, "ny": ny, "nz": nz, "nt": nt}
    return data, dims, coords, meta


# --------------------------
# NetCDF writer
# --------------------------
def to_netcdf_singlevar(
    data: np.ndarray,
    dims: List[str],
    coords: Dict[str, np.ndarray],
    out_path: str,
    var_name: str,
    undef: float,
    *,  
    bin_path_for_time: Optional[str] = None,
):
    """
    Write ndarray to NetCDF file
    """
    if "time" not in dims:
        dims = ["time"] + dims
        data = np.expand_dims(data, axis=0)

    times = guess_time_from_filename(bin_path_for_time) if bin_path_for_time else None
    if times is None:
        times = np.array([np.datetime64("1970-01-01T00:00:00")], dtype="datetime64[ns]")

    da = xr.DataArray(data, dims=dims, name=var_name)
    coord_dict = {k: (k, v) for k, v in coords.items()}
    ds = xr.Dataset({var_name: da}, coords={**coord_dict, "time": ("time", times)})

    ds[var_name].attrs.update({
        "long_name": var_name,
        "missing_value": np.float32(undef),
    })
    if "sigma" in ds.dims:
        ds["sigma"].attrs.update({
            "units": "1",
            "axis": "Z",
        })

    try:
        import netCDF4
        engine = "netcdf4"
        encoding = {var_name: {"zlib": True, "complevel": 4, "_FillValue": np.float32(undef)}}
    except Exception:
        engine = "scipy"
        encoding = {var_name: {"_FillValue": np.float32(undef)}}

    ds.to_netcdf(out_path, engine=engine, encoding=encoding)
    return out_path

def read_first_n_times(data_dir, ctl_path, glob_pattern, n=10, var_name=None):
    ctl = parse_ctl(ctl_path)
    if var_name is None:
        var_name = ctl["vars"][0][0] if ctl.get("vars") else "var"

    files = sorted(data_dir.glob(glob_pattern))[:n]
    all_das = []
    last_meta = None

    for bin_file in files:
        print(f"Reading {bin_file.name}")
        data, dims, coords, meta = read_grads_singlevar_singletime(bin_file, ctl, var_name=var_name)
        last_meta = meta

        if "time" not in dims:
            dims = ["time"] + dims
            data = data[np.newaxis, ...]

        times = guess_time_from_filename(bin_file)
        times = pd.to_datetime(times) if not isinstance(times, pd.DatetimeIndex) else times

        coords_for_xr = {k: v for k, v in coords.items()}
        coords_for_xr["time"] = times

        da = xr.DataArray(data, dims=dims, coords=coords_for_xr, name=var_name)
        all_das.append(da)

    big_da = xr.concat(all_das, dim="time")
    ds = big_da.to_dataset(name=var_name)


    ds[var_name].attrs.setdefault("long_name", var_name)
    ds.attrs.update({
        "title": f"{var_name} first {len(files)} times concatenated from GrADS binaries",
        "history": "time coordinate set from filename",
    })

    return ds
import xarray as xr 
import struct
import numpy as np
import xarray as xr
from typing import Tuple

def _detect_marker(f, expected_payload):
    """
    Detect marker size (4 or 8) and endianness ('>' big, '<' little)
    by peeking the first up-to-8 bytes and comparing to expected_payload.
    """
    pos0 = f.tell()
    head = f.read(8)  # enough to test 4/8 bytes
    f.seek(pos0)

    candidates = [
        ('>i', 4), ('>q', 8),
        ('<i', 4), ('<q', 8),
    ]
    for fmt, msz in candidates:
        if len(head) < msz:
            continue
        val = struct.unpack(fmt, head[:msz])[0]
        if val == expected_payload:
            endian = '>' if fmt[0] == '>' else '<'
            return endian, msz
    raise ValueError(
        "Could not detect record marker size/endianness; "
        "expected payload bytes do not match any 4/8-byte marker."
    )

def compute_zlev(km):
    #import numpy as np
    
    # ----- 入力（Fortranの並びをそのまま転記）-----
    kzd = 100
    
    dz_z = np.array([
        5., 5., 5., 7.5, 7.5, 10., 15., 15., 20., 20.,   # 1..10
        20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,# 11..20
        20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,# 21..30
        25., 25., 30., 30., 40., 40., 50., 50., 60., 80.,# 31..40
        80., 80.,100.,100.,100.,100.,100.,100.,100.,100.,# 41..50
        100.,100.,100.,100.,100.,100.,100.,100.,100.,100.,# 51..60
        100.,200.,200.,200.,200.,200.,200.,200.,200.,200.,# 61..70
        200.,200.,200.,200.,200.,200.,200.,200.,200.,200.,# 71..80
        200.,200.,200.,200.,200.,200.,200.,200.,200.,200.,# 81..90
        200.,200.,200.,200.,200.,200.,200.,200.,200.,200. # 91..100
    ], dtype=np.float64)
        
    assert dz_z.size == kzd
    
    # ----- FortranロジックのPython実装 -----
    # インターフェース（境界）深さ：z_z[0]=0（海面）、以降は厚さの累積で負方向へ
    z_interfaces = np.empty(kzd + 1, dtype=np.float64)
    z_interfaces[0] = 0.0
    z_interfaces[1:] = -np.cumsum(dz_z)
    
    # 層中心深さ：上下の境界の平均（= Fortran: zlev(k) = z_z(k) - dz_z(k)/2）
    zlev = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])
    
    # 例：上位 km 層だけ使うなら
    km_use = km
    z_interfaces_k = z_interfaces[:km_use+1].copy()
    zlev_k    = zlev[:km_use].copy()
    
    #print("top 5 centers (m):", z_centers[:km])      # 例: [-2.5, -7.5, -12.5, -18.75, -26.25]
    #print("top 6 interfaces (m):", z_interfaces[:km+1])# 例: [  0., -5., -10., -15., -22.5, -30. ]
    return zlev_k

def read_basic_to_xarray(
    path: str,
    im: int = 1190, jm: int = 1190, km: int = 47,
    dx: float = 1.0/36.0, dy: float = 1.0/36.0,
#    xlons: float = 117.0, ylats: float = 17.0,
    xlons: float = 119.986111, ylats: float = 16.986111,
    
    
    expected_flag: int = 123456,
) -> Tuple[xr.Dataset, str, int]:
    """
    Read a single Fortran unformatted sequential record made of:
      fildsc(CHAR*4), Z(im,jm,km), ZZ(im,jm,km), DZ(im,jm,km), ichflg(INT32)
    Returns (xarray.Dataset, fildsc, ichflg).
    """

    npts = im * jm * km
    f4 = 4  # float32 bytes
    i4 = 4
    c4 = 4
    payload_expected = c4 + 3 * (npts * f4) + i4

    with open(path, 'rb') as f:
        # 1) detect marker (size & endianness)
        endian, marker_size = _detect_marker(f, payload_expected)

        # 2) read leading marker
        marker_fmt = f"{endian}{'i' if marker_size == 4 else 'q'}"
        nbytes = struct.unpack(marker_fmt, f.read(marker_size))[0]

        if nbytes != payload_expected:
            raise ValueError(f"Record length mismatch: {nbytes} vs expected {payload_expected}")

        # 3) read payload
        payload = f.read(nbytes)
        if len(payload) != nbytes:
            raise EOFError("Unexpected EOF while reading payload.")

        # 4) trailing marker
        nbytes2 = struct.unpack(marker_fmt, f.read(marker_size))[0]
        if nbytes2 != nbytes:
            raise ValueError("Trailing record marker does not match leading marker.")

    # ---- parse payload ----
    off = 0
    # fildsc: first 4 bytes are the CHAR*4 *inside the payload*
    fildsc_bytes = payload[off:off+4]
    fildsc = fildsc_bytes.decode('ascii', errors='replace')
    off += 4

    # big/little for floats follows the detected endianness
    f32 = np.dtype(f"{endian}f4")

    def take_array(count):
        nonlocal off
        view = memoryview(payload)[off:off+count*4]
        arr = np.frombuffer(view, dtype=f32, count=count)
        off += count*4
        return arr

    # Z, ZZ, DZ come in Fortran storage order: i-fastest, then j, then k
    shapeF = (im, jm, km)  # Fortran order
    Z  = take_array(npts).reshape(shapeF, order='F').swapaxes(0, 1)   # -> (jm, im, km)
    ZZ = take_array(npts).reshape(shapeF, order='F').swapaxes(0, 1)
    DZ = take_array(npts).reshape(shapeF, order='F').swapaxes(0, 1)

    # ichflg
    ichflg = struct.unpack(f"{endian}i", payload[off:off+4])[0]
    if ichflg != expected_flag:
        raise ValueError(f"ichflg mismatch: got {ichflg}, expected {expected_flag}")
    off += 4
    if off != len(payload):
        raise ValueError("Byte offset mismatch after parsing payload.")

    # ---- coords & xarray ----
    # x = xlons + dx * np.arange(im, dtype=np.float64)
    # y = ylats + dy * np.arange(jm, dtype=np.float64)
    x = xlons + dx * np.arange(im, dtype=np.float32)
    y = ylats + dy * np.arange(jm, dtype=np.float32)
    z = np.arange(1, km+1, dtype=np.int32)  # level index（必要なら実深度に置換）
    zlev = compute_zlev(km)
    print(z)
    
    ds = xr.Dataset(
        data_vars=dict(
            Z =(("latitude","longitude","z"), Z.astype(np.float64)),
            ZZ=(("latitude","longitude","z"), ZZ.astype(np.float64)),
            DZ=(("latitude","longitude","z"), DZ.astype(np.float64)),
        ),
        coords=dict(
            longitude=("longitude", x, {"long_name": "longitude", "coordinates": "x", "units": "degrees_east"}),
            latitude=("latitude", y, {"long_name": "latitude", "coordinates": "y","units": "degrees_north"}),
            z=("z", z, {"long_name": "z", 
                                 "coordinates": "z", 
                         }),
            zlev=("zlev", zlev, {"long_name": "vertical_level", 
                                 "coordinates": "z", 
                                 "method" : "zlev(k) = z_z(k) - dz_z(k)/2, z_z[0]=0（sea surface)" }),
        ),
        attrs=dict(
            title="basic.dat (Z, ZZ, DZ) from a Fortran unformatted sequential record",
            fortran_unformatted="sequential",
            endianness="big-or-little (auto-detected)",
            record_marker_bytes=marker_size,
            fildsc=fildsc,
            im=im, jm=jm, km=km, dx=dx, dy=dy, xlons=xlons, ylats=ylats,
            ichflg_expected=expected_flag
        )
    )
    return ds, fildsc, ichflg
import numpy as np

def interp_sigma_to_zlev_with_ssh(
    T, Z, ZZ=None, ssh=None, zlev=None,
    grid_h=0, grid_z=1, min_depth=1e-3
):
    """
    Returns: TLEV (ny, nx, nz_out) interpolated array
    - Depth convention: positive downward.
    """
    ny, nx, nz = T.shape
    if ssh is None: 
        ssh = np.zeros((ny, nx), dtype=float)
    if zlev is None: 
        raise ValueError("zlev is required")
    zlev = np.asarray(zlev, dtype=float)
    nz_out = zlev.size

    Zuse = ZZ if (grid_z == 0 and ZZ is not None) else Z
    if Zuse is None:
        raise ValueError("If grid_z=0, provide ZZ; if grid_z=1, provide Z.")

    if grid_h == 2:
        di, dj = -1, 0
    elif grid_h == 3:
        di, dj = 0, -1
    else:
        di, dj = 0, 0

    def nb_idx(ii, jj):
        iim1 = np.clip(ii + di, 0, nx - 1)
        jjm1 = np.clip(jj + dj, 0, ny - 1)
        return iim1, jjm1

    TLEV = np.full((ny, nx, nz_out), np.nan, dtype=float)

    print("Starting interpolation...")
    for j in range(ny):
        if j % max(1, ny // 10) == 0:  # every ~10%
            print(f"Progress: {j}/{ny} rows")

        for i in range(nx):
            im1, jm1 = nb_idx(i, j)

            depth    = Zuse[j, i, -1]
            depth_m1 = Zuse[jm1, im1, -1]

            if (abs(depth) < min_depth) or (abs(depth_m1) < min_depth):
                # shallow/land point
                continue

            r1 = (depth - ssh[j, i]) / depth
            r2 = (depth_m1 - ssh[jm1, im1]) / depth_m1
            SSH_CORR = 0.5 * (ssh[j, i] + ssh[jm1, im1])

            valid_k = np.where(np.isfinite(T[j, i, :]) & np.isfinite(Zuse[j, i, :]))[0]
            if valid_k.size == 0:
                continue
            k_last_defined = valid_k[-1]

            for kk in range(nz_out):
                lev = zlev[kk] - SSH_CORR
                sz_top = 0.5 * (Zuse[j, i, 0] * r1 + Zuse[jm1, im1, 0] * r2)
                if lev > sz_top:
                    v0 = T[j, i, 0]
                    if np.isfinite(v0):
                        TLEV[j, i, kk] = v0
                    continue

                if lev < depth:
                    continue

                filled = False
                for k in range(0, k_last_defined):
                    sz1 = 0.5 * (Zuse[j, i, k] * r1 + Zuse[jm1, im1, k] * r2)
                    sz2 = 0.5 * (Zuse[j, i, k+1] * r1 + Zuse[jm1, im1, k+1] * r2)

                    lo, hi = (sz2, sz1) if sz2 < sz1 else (sz1, sz2)
                    if (lev >= lo) and (lev <= hi):
                        v1, v2 = T[j, i, k], T[j, i, k+1]
                        if np.isfinite(v1) and np.isfinite(v2) and (sz2 != sz1):
                            TLEV[j, i, kk] = v1 + (lev - sz1) * (v2 - v1) / (sz2 - sz1)
                        elif np.isfinite(v1):
                            TLEV[j, i, kk] = v1
                        elif np.isfinite(v2):
                            TLEV[j, i, kk] = v2
                        filled = True
                        break

                if filled:
                    continue

                sz_last = 0.5 * (Zuse[j, i, k_last_defined] * r1 + Zuse[jm1, im1, k_last_defined] * r2)
                if lev <= sz_last:
                    v_last = T[j, i, k_last_defined]
                    if np.isfinite(v_last):
                        TLEV[j, i, kk] = v_last

    print("Interpolation complete.")
    return TLEV
