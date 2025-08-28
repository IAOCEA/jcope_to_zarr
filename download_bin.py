# === BEGIN: téléchargement automatique + décompression dans output_base_dir ===
import requests
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import sys

def _parse_start_datetime(start):
    if isinstance(start, datetime):
        return start
    s = str(start)
    for fmt in ("%Y%m%d%H%M", "%Y%m%d%H", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    raise ValueError(f"start datetime string invalide: {start}")

def fetch_hourly_and_decompress(prefix: str, base_url_dir: str, start, n: int, target_dir: Path,
                                overwrite: bool = False, strict: bool = True, timeout: int = 30):
    """
    Télécharge n fichiers horaires prefix_YYYYMMDDHHMM.bin.gz depuis base_url_dir,
    les décompresse en .bin dans target_dir et renvoie la liste des .bin Path.
    """
    start_dt = _parse_start_datetime(start)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded_bins = []
    for i in range(n):
        dt = start_dt + timedelta(hours=i)
        ts = dt.strftime("%Y%m%d%H%M")  # minutes inclus (tes fichiers ont 0000)
        gz_name = f"{prefix}_{ts}.bin.gz"
        gz_url = f"{base_url_dir.rstrip('/')}/{gz_name}"
        gz_path = target_dir / gz_name
        bin_name = gz_name[:-3]  # retire ".gz"
        bin_path = target_dir / bin_name

        try:
            # 1) Télécharger si nécessaire
            if gz_path.exists() and not overwrite:
                print(f"-> {gz_path.name} exists, skip download")
            else:
                print(f"Downloading: {gz_url}")
                with requests.get(gz_url, stream=True, timeout=timeout) as r:
                    if r.status_code != 200:
                        print(f"ERROR {r.status_code} for {gz_url}", file=sys.stderr)
                        break
                    with open(gz_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=32_768):
                            if chunk:
                                f.write(chunk)

            # 2) Décompresser si nécessaire
            if bin_path.exists() and not overwrite:
                print(f"-> {bin_path.name} exists, skip gunzip")
            else:
                print(f"Decompressing {gz_path.name} -> {bin_path.name}")
                with gzip.open(gz_path, "rb") as f_in, open(bin_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # sanity check
            if bin_path.exists() and bin_path.stat().st_size > 0:
                downloaded_bins.append(bin_path)
            else:
                print(f"WARNING: {bin_path} missing or empty after decompression", file=sys.stderr)
                break

        except Exception as e:
            print(f"Erreur pour {gz_url}: {e}", file=sys.stderr)
            break

    if strict and len(downloaded_bins) != n:
        raise RuntimeError(f"Téléchargés {len(downloaded_bins)}/{n} fichiers pour {prefix} (start={start}).")
    print(f"[OK] {len(downloaded_bins)} fichiers .bin prêts dans {target_dir}")
    return downloaded_bins
