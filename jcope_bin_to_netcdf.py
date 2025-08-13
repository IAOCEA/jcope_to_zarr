#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JCOPE / JAMSTEC Fortran (GrADS) binary -> NetCDF (sigma as-is)

- 読み込み: .ctl から nx, ny, nz, undef, options を取得
- バイナリ: 単一変数・単一時刻前提（TT_, EGT_ など）。sequential(レコードマーカー)対応
- 出力: NetCDF (dims = time, sigma, y, x) ただし nz==1 なら (time, y, x)
- σ座標: ZDEF levels があればそれを sigma 座標に採用（units="1"）
- 欠損: UNDEF を NaN にマスク。_FillValue=UNDEF を付与

使い方例:
  python jcope_bin_to_netcdf.py --ctl TT-2.ctl --bin TT_202102251600.bin -o TT_202102251600.nc
  python jcope_bin_to_netcdf.py --ctl EGT.ctl --bin EGT_202102262300.bin -o EGT_202102262300.nc
"""

import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from datetime import datetime
import re
import numpy as np
from pathlib import Path

def guess_time_from_filename(bin_path: str):
    """
    ファイル名末尾の連続数字から時刻を推定:
      - 12桁: YYYYMMDDHHMM
      - 10桁: YYYYMMDDHH    （分は 00 扱い）
      -  8桁: YYYYMMDD      （時刻は 00:00）
    見つからなければ None を返す。
    """
    name = Path(bin_path).stem
    m = re.search(r'(\d{12})$', name) or re.search(r'(\d{10})$', name) or re.search(r'(\d{8})$', name)
    if not m:
        # 末尾で見つからなければ、文字列内をスキャン
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
    return np.array([np.datetime64(dt)], dtype="datetime64[ns]")  # 長さ1の配列を返す

try:
    import xarray as xr
except Exception as e:
    raise SystemExit("xarray が必要です（pip install xarray）。") from e


# --------------------------
# .CTL パーサ
# --------------------------
def _split(line: str) -> List[str]:
    return re.split(r"\s+", line.strip())

def parse_ctl(ctl_path: str) -> Dict:
    txt = Path(ctl_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    # コメント除去（先頭 * や行内 *... をざっくり）
    lines = [re.sub(r"\*.*$", "", L).strip() for L in txt]
    lines = [L for L in lines if L]

    info = {
        "dset": None,
        "undef": None,
        "options": [],     # e.g., big_endian, little_endian, sequential, yrev, xrev, template ...
        "xdef": None,      # dict: {"n": int, "type": "linear"/"levels", "vals": np.ndarray}
        "ydef": None,
        "zdef": None,
        "tdef": None,      # dict: {"n": int, "type": "linear"/"levels", ...}
        "vars": [],        # list[(name, nlev, desc)]
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
            # 次の行から endvars まで
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
    parts = _split(first_line)
    n = int(parts[1])
    typ = parts[2].lower()
    if typ == "linear":
        v0 = float(parts[3]); dv = float(parts[4])
        vals = v0 + dv * np.arange(n, dtype=np.float64)
        return {"n": n, "type": "linear", "vals": vals}
    elif typ == "levels":
        # levels は複数行にまたがることあり。n 個集める
        remain = " ".join(parts[3:])
        while len(re.findall(r"[-+0-9.eE]+", remain)) < n:
            try:
                remain += " " + next(it)
            except StopIteration:
                break
        nums = re.findall(r"[-+0-9.eE]+", remain)
        vals = np.array(list(map(float, nums[:n])), dtype=np.float64)
        return {"n": n, "type": "levels", "vals": vals}
    else:
        raise ValueError(f"Unsupported {key} type: {typ}")

def _parse_tdef(first_line: str, it) -> Dict:
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
# 補助: 時刻の解釈（最小限）
# --------------------------
_MON = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6,
        "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}

def _parse_grads_time_token(tok: str) -> Optional[datetime]:
    # 例: "00z25feb2021" / "12z01jan2010" / "25feb2021"
    s = tok.strip().lower()
    m = re.match(r"(?:(\d{1,2})z)?(\d{1,2})([a-z]{3})(\d{4})$", s)
    if not m:
        return None
    hh = int(m.group(1)) if m.group(1) else 0
    dd = int(m.group(2))
    mon = _MON.get(m.group(3), 1)
    yy = int(m.group(4))
    try:
        return datetime(yy, mon, dd, hh)
    except Exception:
        return None

def _parse_grads_dt(dt: str) -> Optional[timedelta]:
    # 例: "6hr", "1dy"
    s = dt.strip().lower()
    m = re.match(r"(\d+)([a-z]+)$", s)
    if not m:
        return None
    val = int(m.group(1)); unit = m.group(2)
    if unit in ("hr", "hour", "hours"):
        return timedelta(hours=val)
    if unit in ("mn", "min", "mins", "minute", "minutes"):
        return timedelta(minutes=val)
    if unit in ("dy", "day", "days"):
        return timedelta(days=val)
    # "mo","yr" はばらつくので厳密対応は省略
    return None


# --------------------------
# バイナリ読み出し
# --------------------------
def _dtype_from_options(options: List[str]) -> np.dtype:
    # 通常は float32。big/little は options に従う。
    endian = ">" if "big_endian" in options else "<" if "little_endian" in options else ">"
    return np.dtype(endian + "f4")

def _read_sequential_records(fpath: str, nrec: int, count_per_rec: int, dtype: np.dtype) -> np.ndarray:
    """Fortran unformatted sequential: 各レコード
       [4byte len][payload][4byte len] の並び。len は payload バイト長。
       ここでは record marker は little-endian int32 と仮定（一般的な x86/gfortran）"""
    out = np.empty((nrec, count_per_rec), dtype=dtype.newbyteorder("="))  # payload は dtype のエンディアン
    with open(fpath, "rb") as f:
        for i in range(nrec):
            # 先頭マーカー
            b = f.read(4)
            if len(b) != 4:
                raise EOFError(f"Sequential read: unexpected EOF at record {i} (lead).")
            reclen_le = int(np.frombuffer(b, dtype="<i4")[0])

            # ペイロード
            payload = f.read(reclen_le)
            if len(payload) != reclen_le:
                raise EOFError(f"Sequential read: short payload at record {i}.")

            # 末尾マーカー
            trail = f.read(4)
            if len(trail) != 4:
                raise EOFError(f"Sequential read: unexpected EOF at record {i} (trail).")

            arr = np.frombuffer(payload, dtype=dtype, count=count_per_rec)
            if arr.size != count_per_rec:
                raise ValueError(
                    f"Record {i}: expected {count_per_rec} values, got {arr.size}. "
                    f"(dtype={dtype}, reclen={reclen_le})"
                )
            out[i, :] = arr
    return out

def read_grads_singlevar_singletime(
    bin_path: str,
    ctl: Dict,
    var_name: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], Dict]:
    """
    単一変数・単一時刻のファイルを読み込み、(sigma?, y, x) あるいは (y, x) の ndarray を返す。
    返値:
      data : ndarray  shape = (nz, ny, nx) or (ny, nx)
      dims : list[str]  ["sigma","y","x"] or ["y","x"]
      coords : dict     {"sigma":..., "y":..., "x":...}
      meta : dict       {"undef": float, "options": list, ...}
    """
    nx = ctl["xdef"]["n"]; ny = ctl["ydef"]["n"]
    nz = ctl["zdef"]["n"] if ctl.get("zdef") else 1  # ZDEF 無ければ 2D とみなす
    nt = ctl["tdef"]["n"] if ctl.get("tdef") else 1

    if nt != 1:
        # 単一時刻想定だが、linear の場合はとりあえず最初の1だけ読む…などの分岐も可
        # ここでは明示的に警告しておく
        print(f"[warn] TDEF n={nt}. 本スクリプトは単一時刻想定です。")

    options = ctl.get("options", [])
    dtype = _dtype_from_options(options)
    undef = ctl.get("undef", 1.0e20)

    # x, y, sigma 座標値
    x_vals = ctl["xdef"]["vals"]
    y_vals = ctl["ydef"]["vals"]

    if ctl.get("zdef"):
        if ctl["zdef"]["type"] == "levels":
            sigma_vals = ctl["zdef"]["vals"]
        else:
            # linear の σ は珍しいが、いちおう値列を作る
            sigma_vals = ctl["zdef"]["vals"]
    else:
        sigma_vals = None  # 2D

    sequential = "sequential" in options

    # 読む総数: nt * nz * ny * nx（単一時刻・単一変数）
    n_per_level = nx * ny
    n_levels = nz
    expected_vals = n_levels * n_per_level

    if not sequential:
        a = np.fromfile(bin_path, dtype=dtype)
        if a.size != expected_vals:
            raise ValueError(
                f"サイズ不一致: file has {a.size}, expected {expected_vals} (= {nz}*{ny}*{nx}). "
                f"endianness / sequential / 形状を確認してください。"
            )
        # GrADS/Fortran の典型: x が最速。ここでは (nx, ny, nz) を組んでから (nz, ny, nx) に転置
        if nz == 1:
            arr = a.reshape((nx, ny), order="F").T                 # (ny, nx)
            dims = ["y", "x"]; coords = {"y": y_vals, "x": x_vals}
        else:
            arr = a.reshape((nx, ny, nz), order="F").transpose(2, 1, 0)  # (nz, ny, nx)
            dims = ["sigma", "y", "x"]; coords = {"sigma": sigma_vals, "y": y_vals, "x": x_vals}
    else:
        # sequential: 1 レコード = 1 xy 面（= nx*ny 値）と仮定し、nz 回読む
        recs = _read_sequential_records(bin_path, nrec=n_levels, count_per_rec=n_per_level, dtype=dtype)
        if nz == 1:
            arr_xy = recs[0, :]
            arr = arr_xy.reshape((nx, ny), order="F").T
            dims = ["y", "x"]; coords = {"y": y_vals, "x": x_vals}
        else:
            # (nz, ny, nx)
            arr = np.vstack([recs[k, :].reshape((nx, ny), order="F").T[np.newaxis, ...] for k in range(nz)])
            dims = ["sigma", "y", "x"]; coords = {"sigma": sigma_vals, "y": y_vals, "x": x_vals}

    # 欠損マスク
    data = np.where(np.abs(arr) >= 0.9 * float(undef), np.nan, arr).astype(np.float32)

    # xrev/yrev は座標値側で降順になるだけなので、必要なら反転（任意）
    if "yrev" in options:
        data = data[..., ::-1, :]
        coords["y"] = coords["y"][::-1]
    if "xrev" in options:
        data = data[..., ::-1]
        coords["x"] = coords["x"][::-1]

    meta = {"undef": float(undef), "options": options, "nx": nx, "ny": ny, "nz": nz, "nt": nt}
    return data, dims, coords, meta


# --------------------------
# NetCDF へ保存
# --------------------------
def to_netcdf_singlevar(
    data: np.ndarray,
    dims: List[str],
    coords: Dict[str, np.ndarray],
    out_path: str,
    var_name: str,
    undef: float,
    tdef: Optional[Dict] = None,
    *,  # 以降はキーワード専用
    bin_path_for_time: Optional[str] = None,
):
    # time 次元を必ず作る（単一時刻）
    if "time" not in dims:
        dims = ["time"] + dims
        data = np.expand_dims(data, axis=0)

    # --- 時刻: ファイル名から 1 本だけ推定。失敗時は 1970-01-01 にフォールバック ---
    if bin_path_for_time is not None:
        times = guess_time_from_filename(bin_path_for_time)
    else:
        times = None
    if times is None:
        times = np.array([np.datetime64("1970-01-01T00:00:00")], dtype="datetime64[ns]")

    # DataArray & coords
    da = xr.DataArray(data, dims=dims, name=var_name)
    coord_dict = {k: (k, v) for k, v in coords.items()}
    ds = xr.Dataset({var_name: da}, coords={**coord_dict, "time": ("time", times)})

    # 変数属性
    ds[var_name].attrs.update({
        "long_name": var_name,
        "missing_value": np.float32(undef),
 #       "_FillValue": np.float32(undef),
    })
    if "sigma" in ds.dims:
        ds["sigma"].attrs.update({
            "units": "1",
            "axis": "Z",
        })
    # （任意）もし誤って attrs に _FillValue が乗っていたら外す
    ds[var_name].attrs.pop("_FillValue", None)


    # グローバル属性にメモ（TDEF は使っていない旨）
    ds.attrs.update({
        "title": f"{var_name} exported from GrADS binary (sigma kept as-is)",
        "history": "time coordinate set from filename; TDEF ignored for coord length",
    })

    # 書き出し（netCDF4 があれば圧縮、無ければ非圧縮）
    try:
        import netCDF4  # noqa
        engine = "netcdf4"
        encoding = {var_name: {"zlib": True, "complevel": 4, "_FillValue": np.float32(undef)}}
    except Exception:
        engine = "scipy"
        encoding = {var_name: {"_FillValue": np.float32(undef)}}

    ds.to_netcdf(out_path, engine=engine, encoding=encoding)




# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="GrADS binary (single var, single time) -> NetCDF (sigma as-is)")
    ap.add_argument("--ctl", required=True, help=".ctl file path")
    ap.add_argument("--bin", required=True, help="binary file path (TT_*.bin, EGT_*.bin, etc.)")
    ap.add_argument("-o", "--out", required=False, help="output NetCDF path")
    ap.add_argument("--var", default=None, help="variable name for the output (default: from file or .ctl)")
    args = ap.parse_args()

    ctl = parse_ctl(args.ctl)
    var_name = args.var
    if var_name is None:
        # .ctl の VARS に1つだけあるならそれ、無ければファイル名先頭（_ 区切り）
        if ctl["vars"]:
            var_name = ctl["vars"][0][0]
        else:
            var_name = Path(args.bin).stem.split("_")[0]

    data, dims, coords, meta = read_grads_singlevar_singletime(args.bin, ctl, var_name=var_name)

    out_path = args.out or (str(Path(args.bin).with_suffix(".nc")))
    to_netcdf_singlevar(
        data=data,
        dims=dims,
        coords=coords,
        out_path=out_path,
        var_name=var_name,
        undef=meta["undef"],
        tdef=ctl.get("tdef"),
        bin_path_for_time=args.bin,  # ★ 追加：ファイル名から時刻を作る
    )
    print(f"[OK] wrote: {out_path}")
    print(f" shape: {data.shape}, dims: {dims}")
    print(f" stats: min={np.nanmin(data):.6g}, max={np.nanmax(data):.6g}, mean={np.nanmean(data):.6g}")
    print(f" meta : {meta}")

if __name__ == "__main__":
    main()
