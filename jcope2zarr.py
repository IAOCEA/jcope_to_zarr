#!/usr/bin/env python3
import argparse
import gzip
import io
import json
import re
import struct
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from numcodecs import Blosc

# ====== CTL parsing ======
CTL_VAR_BLOCK_RE = re.compile(r"^\s*vars\s+(\d+)\s*$", re.IGNORECASE)
CTL_ENDVARS_RE = re.compile(r"^\s*endvars\s*$", re.IGNORECASE)

# 追加：候補 2D 形状の生成（n = 要素数、CTL の比 ny/nx をヒントに並べ替え）
def _factor_pairs(n: int, nx_hint: int, ny_hint: int, max_pairs: int = 200):
    import math
    pairs = []
    r_hint = (ny_hint / nx_hint) if (nx_hint and ny_hint) else None
    # すべての約数ペア（ny',nx'）を列挙
    root = int(math.sqrt(n))
    for d in range(1, root+1):
        if n % d == 0:
            ny_p, nx_p = d, n // d
            pairs.append((ny_p, nx_p))
            if ny_p != nx_p:
                pairs.append((nx_p, ny_p))
            if len(pairs) > 5_000:  # 念のため上限
                break
    # CTL のアスペクト比に近い順 → 正方に近い順 → 小さい方が先
    def score(p):
        ny_p, nx_p = p
        r = ny_p / nx_p if nx_p else 1.0
        s1 = abs((r_hint - r)) if r_hint else abs(1.0 - r)
        s2 = abs(ny_p*nx_p - n)
        s3 = min(ny_p, nx_p)
        return (s1, s2, -s3)
    pairs.sort(key=score)
    return pairs[:max_pairs]


# 置換：Basic.dat の 1 レコードを複数平面・多 dtype・多エンディアンで解釈
def reshape_candidates(buf: bytes, ny: int, nx: int):
    """
    k*(ny*nx) 要素を許容。dtype/endian を広く探索し、k面に分割して返す。
    試す型：
      float32/float64/int32/int16 × little/big endian
    """
    dtypes = [
        np.dtype('<f4'), np.dtype('>f4'),
        np.dtype('<f8'), np.dtype('>f8'),
        np.dtype('<i4'), np.dtype('>i4'),
        np.dtype('<i2'), np.dtype('>i2'),
    ]
    base = ny * nx
    for dt in dtypes:
        item = dt.itemsize
        if len(buf) % item != 0:
            continue
        arr1d = np.frombuffer(buf, dtype=dt)
        if base > 0 and arr1d.size % base == 0:
            k = arr1d.size // base
            planes = arr1d.reshape((k, ny, nx))
            tag = f"{dt.kind}{item*8}{'le' if dt.byteorder in ('<','=', '|') else 'be'}"
            for iz in range(k):
                yield (tag, f"z{iz}", planes[iz])


# 置換：build_static_from_basic_and_ctls の中で、一次候補が空のときに
#       形状推定（因数分解）まで自動で試みるフォールバックを追加
def build_static_from_basic_and_ctls(basic_path: str, nx: int, ny: int, any_ctl_meta: dict) -> xr.Dataset:
    recs = autodetect_fortran_records(basic_path)
    arrays = []; scores = []

    # --- まずは (ny,nx) 固定で解釈してみる ---
    for i, rec in enumerate(recs):
        found = False
        for dt, od, arr in reshape_candidates(rec, ny, nx):
            name = f"var_{i:02d}_{dt}{od}"
            arrays.append((name, arr))
            scores.append((name, score_bathy(arr)))
            found = True
        # 何も見つからなければ後でフォールバック
    # --- ここまでで見つかれば採用 ---
    if arrays:
        best_name, best_score = max(scores, key=lambda t: t[1])
    else:
        # ===== フォールバック =====
        # レコードごとに「総要素数」を出し、それを因数分解して (ny', nx') を探索。
        # CTL のアスペクト比に近い形状から試す。dtype/endian も総当たり。
        dtypes = [
            np.dtype('<f4'), np.dtype('>f4'),
            np.dtype('<f8'), np.dtype('>f8'),
            np.dtype('<i4'), np.dtype('>i4'),
            np.dtype('<i2'), np.dtype('>i2'),
        ]
        for i, rec in enumerate(recs):
            for dt in dtypes:
                item = dt.itemsize
                if len(rec) % item != 0:
                    continue
                arr1d = np.frombuffer(rec, dtype=dt)
                n = arr1d.size
                for ny_p, nx_p in _factor_pairs(n, nx, ny, max_pairs=200):
                    try:
                        arr2d = arr1d.reshape((ny_p, nx_p))
                    except Exception:
                        continue
                    tag = f"{dt.kind}{item*8}{'le' if dt.byteorder in ('<','=', '|') else 'be'}"
                    name = f"var_{i:02d}_{tag}_auto{ny_p}x{nx_p}"
                    arrays.append((name, arr2d))
                    scores.append((name, score_bathy(arr2d)))
            # 1 レコードから大量に生成されるので、適度に打ち切り
            if len(arrays) > 500:
                break
        if not arrays:
            raise ValueError(
                "No ny×nx-compatible records in Basic.dat; "
                "also failed to infer shape by factorization. "
                "（Basic.dat の解像度が CTL と異なる可能性。ctl/Basic.dat の組合せを確認してください）"
            )
        best_name, best_score = max(scores, key=lambda t: t[1])

    # --- Dataset 構築（最終的に CTL 側の ny,nx を軸に採用） ---
    y = np.arange(ny, dtype=np.int32); x = np.arange(nx, dtype=np.int32)
    ds = xr.Dataset(coords={"y": ("y", y), "x": ("x", x)})
    for name, arr in arrays:
        # 形状が (ny,nx) と違う推定品は、添え字座標で保持しない（混乱回避のためスキップ）。
        # もし残したい場合はここで regrid/resize してください。
        if arr.shape == (ny, nx):
            ds[name] = xr.DataArray(arr, dims=("y", "x"), name=name)

    # Hg 自動検出
    if ds.data_vars:
        # scores のうち、(ny,nx) に一致したものだけで再算定
        filtered = [(n, s) for (n, s) in scores if n in ds.data_vars]
        if filtered:
            best_name, best_score = max(filtered, key=lambda t: t[1])
            if best_score > 0:
                ds = ds.rename({best_name: "Hg"})
                ds["Hg"].attrs.update({"units": "m", "positive": "down",
                                       "long_name": "bathymetry", "note": f"auto from {best_name} score={best_score:.3f}"})

    # lon/lat（2D）を CTL から
    def axis_from_def(defn, n):
        if not defn: return None
        if defn[0] == "linear":
            start, inc = defn[1], defn[2]
            return start + inc*np.arange(n)
        elif defn[0] == "levels":
            vals = defn[1]
            if len(vals) >= n: return np.array(vals[:n])
        return None

    xs = axis_from_def(any_ctl_meta.get("xdef"), nx)
    ys = axis_from_def(any_ctl_meta.get("ydef"), ny)
    if xs is not None and ys is not None:
        X, Y = np.meshgrid(xs, ys)
        ds["lon"] = xr.DataArray(X, dims=("y","x"))
        ds["lat"] = xr.DataArray(Y, dims=("y","x"))
        ds["lon"].attrs["standard_name"] = "longitude"
        ds["lat"].attrs["standard_name"] = "latitude"

    # 後続 append 用に CTL メタを保存
    ds.attrs["ctl_meta_example"] = json.dumps(any_ctl_meta)
    ds.attrs["source_basic"] = basic_path
    return ds

def parse_ctl(path: str) -> dict:
    nx = ny = None
    xdef = ydef = zdef = None
    options = set()
    vars_map: Dict[str, dict] = {}
    in_vars = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            low = line.lower()
            if not line or line.startswith(("*", "!", "#")):
                continue
            if low.startswith("options"):
                for opt in re.split(r"\s+", low)[1:]:
                    if opt: options.add(opt)
                continue
            if low.startswith("xdef"):
                toks = re.split(r"\s+", low)
                nx = int(float(toks[1]))
                kind = toks[2]
                xdef = ("linear", float(toks[3]), float(toks[4])) if kind=="linear" else ("levels", [float(t) for t in toks[3:]])
                continue
            if low.startswith("ydef"):
                toks = re.split(r"\s+", low)
                ny = int(float(toks[1]))
                kind = toks[2]
                ydef = ("linear", float(toks[3]), float(toks[4])) if kind=="linear" else ("levels", [float(t) for t in toks[3:]])
                continue
            if low.startswith("zdef"):
                toks = re.split(r"\s+", low)
                nz = int(float(toks[1]))
                kind = toks[2]
                zdef = ("linear", nz, float(toks[3]), float(toks[4])) if kind=="linear" else ("levels", nz, [float(t) for t in toks[3:]])
                continue
            m = CTL_VAR_BLOCK_RE.match(low)
            if m:
                in_vars = True
                continue
            if CTL_ENDVARS_RE.match(low):
                in_vars = False
                continue
            if in_vars:
                toks = re.split(r"\s+", line)
                if len(toks) >= 3:
                    vname = toks[0]
                    try: nlev = int(float(toks[1]))
                    except Exception: nlev = 1
                    units = toks[2]
                    desc = " ".join(toks[3:]) if len(toks) > 3 else ""
                    vars_map[vname] = {"nlev": nlev, "units": units, "desc": desc}
    return {"nx": nx, "ny": ny, "xdef": xdef, "ydef": ydef, "zdef": zdef, "options": list(options), "vars": vars_map, "path": str(path)}

def infer_nx_ny_from_ctls(ctl_paths: List[str]) -> Tuple[int, int, Dict[str, dict]]:
    parsed: Dict[str, dict] = {}
    nxny = set()
    for p in ctl_paths:
        meta = parse_ctl(p)
        parsed[p] = meta
        if meta["nx"] and meta["ny"]:
            nxny.add((meta["nx"], meta["ny"]))
    if not nxny:
        raise ValueError("No nx,ny found from CTLs")
    if len(nxny) > 1:
        raise ValueError(f"Inconsistent nx,ny across CTLs: {nxny}")
    (nx, ny) = list(nxny)[0]
    return nx, ny, parsed

# ====== Fortran unformatted ======
def read_fortran_unformatted_records(fileobj: io.BufferedReader, marker_size: int, endian: str):
    fmt_int = ("<i" if endian=="little" else ">i") if marker_size==4 else ("<q" if endian=="little" else ">q")
    while True:
        head = fileobj.read(marker_size)
        if not head: break
        if len(head) != marker_size: raise IOError("record header truncated")
        (nbytes,) = struct.unpack(fmt_int, head)
        payload = fileobj.read(nbytes)
        if len(payload) != nbytes: raise IOError("record payload truncated")
        tail = fileobj.read(marker_size)
        if len(tail) != marker_size: raise IOError("record tail truncated")
        (nbytes2,) = struct.unpack(fmt_int, tail)
        if nbytes2 != nbytes: raise IOError("record markers mismatch")
        yield payload

def autodetect_fortran_records(path: str):
    for msize, endian in [(4,"little"), (4,"big"), (8,"little"), (8,"big")]:
        try:
            with open(path, "rb") as f:
                return list(read_fortran_unformatted_records(f, msize, endian))
        except Exception:
            continue
    raise ValueError(f"Failed to parse Fortran unformatted: {path}")

# ====== Basic.dat -> static Dataset ======
def reshape_candidates_bad(buf: bytes, ny: int, nx: int):
    """k*(ny*nx) 要素を許容。f32/f64/i32/i16 を試し、k面に分割。"""
    for dtype in (np.float32, np.float64, np.int32, np.int16):
        item = np.dtype(dtype).itemsize
        if len(buf) % item != 0: continue
        arr1d = np.frombuffer(buf, dtype=dtype)
        base = ny*nx
        if base <= 0 or arr1d.size % base != 0: continue
        k = arr1d.size // base
        planes = arr1d.reshape((k, ny, nx))
        tag = {np.float32:"f32", np.float64:"f64", np.int32:"i32", np.int16:"i16"}[dtype]
        for iz in range(k):
            yield (tag, f"z{iz}", planes[iz])

def score_bathy(a: np.ndarray) -> float:
    valid = np.isfinite(a)
    if valid.mean() < 0.9: return -np.inf
    vals = a[valid]
    pos = (vals > 0).mean()
    too_high = (vals > 12000).mean()
    med = np.nanmedian(vals)
    med_ok = 1.0 - min(abs(med-3000)/6000.0, 1.0)
    return 0.6*pos + 0.3*(1.0 - too_high) + 0.1*med_ok

def is_mask_like(a: np.ndarray) -> bool:
    valid = np.isfinite(a)
    if valid.mean() < 0.9: return False
    vals = a[valid]
    uniq = np.unique(np.round(vals, 6))
    if uniq.size <= 4: return True
    frac_zeroish = (np.abs(vals) < 1e-6).mean()
    frac_pos = (vals > 0).mean()
    return (frac_zeroish > 0.2 and frac_pos > 0.2)

def xy_chunks(ny: int, nx: int, target: int = 1_000_000):
    xchunk = min(nx, max(1, target // max(1, ny)))
    ychunk = min(ny, max(1, target // max(1, xchunk)))
    while xchunk * ychunk > target and xchunk > 1:
        xchunk //= 2
    return ychunk, xchunk

def axis_from_def(defn, n):
    if not defn: return None
    if defn[0] == "linear":
        start, inc = defn[1], defn[2]
        return start + inc*np.arange(n)
    elif defn[0] == "levels":
        vals = defn[1]
        if len(vals) >= n: return np.array(vals[:n])
    return None

def build_static_from_basic_and_ctls_bad(basic_path: str, nx: int, ny: int, any_ctl_meta: dict) -> xr.Dataset:
    recs = autodetect_fortran_records(basic_path)
    arrays = []; scores = []
    for i, rec in enumerate(recs):
        found = False
        for dt, od, arr in reshape_candidates(rec, ny, nx):
            name = f"var_{i:02d}_{dt}{od}"
            arrays.append((name, arr))
            scores.append((name, score_bathy(arr)))
            found = True
        if not found:
            pass
    if not arrays:
        raise ValueError("No ny×nx-compatible records in Basic.dat")

    best_name, best_score = max(scores, key=lambda t: t[1])
    y = np.arange(ny, dtype=np.int32); x = np.arange(nx, dtype=np.int32)
    ds = xr.Dataset(coords={"y": ("y", y), "x": ("x", x)})
    for name, arr in arrays:
        ds[name] = xr.DataArray(arr, dims=("y", "x"), name=name)

    if best_score > 0:
        ds = ds.rename({best_name: "Hg"})
        ds["Hg"].attrs.update({"units": "m", "positive": "down", "long_name": "bathymetry", "note": f"auto from {best_name} score={best_score:.3f}"})

    # 2D lon/lat を CTL から
    xs = axis_from_def(any_ctl_meta.get("xdef"), nx)
    ys = axis_from_def(any_ctl_meta.get("ydef"), ny)
    if xs is not None and ys is not None:
        X, Y = np.meshgrid(xs, ys)
        ds["lon"] = xr.DataArray(X, dims=("y","x"))
        ds["lat"] = xr.DataArray(Y, dims=("y","x"))
        ds["lon"].attrs["standard_name"] = "longitude"
        ds["lat"].attrs["standard_name"] = "latitude"

    # 後続 append 用に CTL メタを保存
    ds.attrs["ctl_meta_example"] = json.dumps(any_ctl_meta)
    ds.attrs["source_basic"] = basic_path
    return ds

# ====== BIN reading ======
TS_RE = re.compile(r"(\d{10,14})")

def parse_time_from_filename(name: str) -> np.datetime64:
    m = TS_RE.search(name)
    if not m: return np.datetime64("NaT")
    ts = m.group(1)
    fmt = "%Y%m%d"
    if len(ts) >= 10: fmt = "%Y%m%d%H"
    if len(ts) >= 12: fmt = "%Y%m%d%H%M"
    from datetime import datetime
    return np.datetime64(datetime.strptime(ts, fmt))

def read_fortran_seq_levels(raw: bytes, nx: int, ny: int, nlev: int) -> np.ndarray:
    buf = io.BytesIO(raw)
    for msize, endian in [(4,"little"), (4,"big"), (8,"little"), (8,"big")]:
        try:
            recs = list(read_fortran_unformatted_records(buf, msize, endian))
            if len(recs) == nlev:
                stacks = []
                for rec in recs:
                    a = np.frombuffer(rec, dtype=np.float32)
                    if a.size != nx*ny: raise ValueError("rec size mismatch")
                    stacks.append(a.reshape((ny, nx)))
                return np.stack(stacks, axis=0)
        except Exception:
            buf.seek(0)
            continue
    raise ValueError("could not parse sequential levels")

def read_bin_as_da(bin_path: str, meta: dict, var_stem: str, nx: int, ny: int) -> xr.DataArray:
    vars_map = meta.get("vars", {})
    options = set(meta.get("options", []))
    nlev = None
    if var_stem in vars_map:
        nlev = vars_map[var_stem]["nlev"]
    elif vars_map:
        nlev = sorted([v["nlev"] for v in vars_map.values()], reverse=True)[0]
    if not nlev: nlev = 1

    if bin_path.endswith(".gz"):
        with gzip.open(bin_path, "rb") as f: raw = f.read()
    else:
        with open(bin_path, "rb") as f: raw = f.read()

    if "sequential" in options:
        data = read_fortran_seq_levels(raw, nx, ny, nlev)
    else:
        arr = np.frombuffer(raw, dtype=np.float32)
        total = arr.size
        expect = nx*ny*nlev
        if total != expect:
            if nx*ny == 0 or total % (nx*ny) != 0:
                nz = max(1, total // (nx * max(1, ny)))
                data = arr[: nz*ny*nx].reshape((nz, ny, nx))
            else:
                nz = total // (nx*ny)
                data = arr.reshape((nz, ny, nx))
            nlev = data.shape[0]
        else:
            data = arr.reshape((nlev, ny, nx))

    z = np.arange(nlev, dtype=np.int16)
    da = xr.DataArray(data, dims=("z","y","x"), coords={"z": z}, name=var_stem)
    if var_stem in vars_map:
        da.attrs["units"] = vars_map[var_stem].get("units","")
        if vars_map[var_stem].get("desc"): da.attrs["long_name"] = vars_map[var_stem]["desc"]
    return da

# ====== Zarr helpers ======
COMP = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

def ensure_zarr_initialized(out_store: str, static_ds: xr.Dataset, xy_chunks: Tuple[int,int]):
    p = Path(out_store)
    if p.exists(): return
    enc = {v: {"chunks": xy_chunks, "compressor": COMP} for v in static_ds.data_vars}
    static_ds.to_zarr(out_store, mode="w", encoding=enc)

# 置換してください（元の append_one を丸ごと置き換え）
def append_one(out_store: str, da: xr.DataArray, tstamp: np.datetime64,
               xy_chunks: Tuple[int,int], zchunk: int=10):
    """
    最初の追記では append_dim を使わずに 'time' 次元を作成。
    2回目以降は append_dim='time' で縦に伸ばす。
    """
    # (time, z?, y, x) に拡張
    ds_to_write = da.expand_dims(time=[tstamp]).to_dataset(name=da.name)

    # チャンク & 圧縮
    if "z" in da.dims:
        enc = {da.name: {"chunks": (1, zchunk, xy_chunks[0], xy_chunks[1]),
                         "compressor": COMP}}
    else:
        enc = {da.name: {"chunks": (1, xy_chunks[0], xy_chunks[1]),
                         "compressor": COMP}}

    # 既存ストアに time 次元があるか？
    need_create_time = True
    if Path(out_store).exists():
        try:
            ds0 = xr.open_zarr(out_store)
            # 次元サイズの取得は .sizes を使う
            need_create_time = ("time" not in ds0.sizes)
            ds0.close()
        except Exception:
            need_create_time = True

    if need_create_time:
        # 最初の1回：append_dim なしで書き込み（time 次元を新規作成）
        ds_to_write.to_zarr(out_store, mode="a", encoding=enc)
    else:
        # 2回目以降：time 次元へ追記
        ds_to_write.to_zarr(out_store, mode="a", append_dim="time", encoding=enc)

def append_one_bad (out_store: str, da: xr.DataArray, tstamp: np.datetime64, xy_chunks: Tuple[int,int], zchunk: int=10):
    da = da.expand_dims(time=[tstamp])
    enc = {da.name: {"chunks": (1, zchunk if "z" in da.dims else 1, xy_chunks[0], xy_chunks[1]) if "z" in da.dims
                     else (1, xy_chunks[0], xy_chunks[1]),
                     "compressor": COMP}}
    da.to_dataset(name=da.name).to_zarr(out_store, mode="a", append_dim="time", encoding=enc)

# ====== CLI ======
def main():
    ap = argparse.ArgumentParser(description="Init Zarr from Basic.dat + CTLs, then append BINs" )
    ap.add_argument("bins", nargs="*", help="Input .bin/.bin.gz (glob allowed)")
    ap.add_argument("--ctl-dir", default="./ctl", help="Directory with CTL files (default: ./ctl)")
    ap.add_argument("--basic", default=None, help="Path to Basic.dat (default: <ctl-dir>/Basic.dat)")
    ap.add_argument("--out", default="jcope.zarr", help="Output Zarr store path (default: jcope.zarr)")
    ap.add_argument("--append", action="store_true", help="Append BINs along time (informational)")
    ap.add_argument("--time-source", choices=["filename","ctl"], default="filename")
    args = ap.parse_args()

    out_store = args.out
    basic_path = args.basic or str(Path(args.ctl_dir) / "Basic.dat")  # default ./ctl/Basic.dat

    # 既存Zarrのメタ（CTLなど）を読む
    z_existing_meta = None
    if Path(out_store).exists():
        try:
            ds0 = xr.open_zarr(out_store)
            nx, ny = int(ds0.sizes["x"]), int(ds0.sizes["y"])
            if "ctl_meta_example" in ds0.attrs:
                z_existing_meta = json.loads(ds0.attrs["ctl_meta_example"])
            ds0.close()
        except Exception:
            z_existing_meta = None

    # 初期化
    if not Path(out_store).exists():
        ctl_paths = sorted(Path(args.ctl_dir).glob("*.ctl"))
        if not ctl_paths:
            raise SystemExit(f"No .ctl found in {args.ctl_dir}")
        nx, ny, parsed_ctls = infer_nx_ny_from_ctls([str(p) for p in ctl_paths])
        any_ctl_meta = next(iter(parsed_ctls.values()))
        if not Path(basic_path).exists():
            raise SystemExit(f"Basic file not found: {basic_path}")
        static_ds = build_static_from_basic_and_ctls(basic_path, nx=nx, ny=ny, any_ctl_meta=any_ctl_meta)
        ychunk, xchunk = xy_chunks(ny, nx, target=1_000_000)
        ensure_zarr_initialized(out_store, static_ds, (ychunk, xchunk))
        print(f"[init] {out_store} created with static fields, lon/lat, chunks y={ychunk} x={xchunk}")
    else:
        ds0 = xr.open_zarr(out_store)
        nx, ny = int(ds0.sizes["x"]), int(ds0.sizes["y"])
        ds0.close()
        ychunk, xchunk = xy_chunks(ny, nx, target=1_000_000)

    # BIN 追記
    bin_list: List[str] = []
    for p in args.bins:
        bin_list.extend(sorted(glob(p)))
    if not bin_list:
        print("[init-only] Zarr ready.")
        return

    for b in bin_list:
        bname = Path(b).name
        stem = bname.split("_")[0].split(".")[0]
        ctl_meta = None
        ctl_path = Path(args.ctl_dir) / f"{stem}.ctl"
        if ctl_path.exists():
            ctl_meta = parse_ctl(str(ctl_path))
        elif z_existing_meta is not None:
            ctl_meta = z_existing_meta
        else:
            cands = sorted(Path(args.ctl_dir).glob("*.ctl"))
            if cands:
                ctl_meta = parse_ctl(str(cands[0]))
        if ctl_meta is None:
            raise SystemExit("No CTL metadata available (provide --ctl-dir or ensure Zarr contains ctl_meta_example).")

        da = read_bin_as_da(b, ctl_meta, stem, nx=nx, ny=ny)
        tstamp = parse_time_from_filename(bname)
        if str(tstamp) == "NaT":
            if args.time_source == "ctl":
                tstamp = np.datetime64("now")
            else:
                raise ValueError(f"Cannot infer time from filename: {bname}")
        append_one(out_store, da, tstamp, (ychunk, xchunk), zchunk=10)
        print(f"[append] {bname} time={tstamp} shape={tuple(da.shape)} var={da.name}")

if __name__ == "__main__":
    main()
