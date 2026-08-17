"""Download NIH ChestX-ray14 metadata and image tars.

Images are extracted, resized to `image_size`, saved as JPEG, then the
original 1024 px PNG and the tar are deleted. That is the only way the
full 112k-image set fits on a typical 8 GB laptop disk.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from cxr.constants import NIH_IMAGE_TARS, NIH_METADATA_URLS


def _http_get(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cxr-diagnostic/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            bar.update(len(chunk))
        bar.close()
    tmp.replace(dest)


def download_first(urls: list[str], dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    errors: list[str] = []
    for url in urls:
        try:
            print(f"Downloading {dest.name} from {url}")
            _http_get(url, dest)
            if dest.exists() and dest.stat().st_size > 100:
                return dest
        except Exception as exc:  # noqa: BLE001 - we try the next mirror
            errors.append(f"{url} -> {exc}")
            if dest.exists():
                dest.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {dest.name}:\n" + "\n".join(errors))


def download_metadata(meta_dir: Path) -> dict[str, Path]:
    meta_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, urls in NIH_METADATA_URLS.items():
        dest = meta_dir / name
        try:
            download_first(urls, dest)
            paths[name] = dest
        except RuntimeError as exc:
            print(f"[warn] {exc}")
    return paths


def _save_resized(src: Path, dest: Path, image_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("L")
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        img.save(dest, format="JPEG", quality=92, optimize=True)


def process_tar(
    tar_path: Path,
    image_dir: Path,
    image_size: int = 224,
    keep_tar: bool = False,
) -> int:
    """Extract one NIH tar, resize every PNG, delete originals. Returns count."""
    count = 0
    with tarfile.open(tar_path, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".png")]
        for member in tqdm(members, desc=f"resize {tar_path.name}"):
            name = Path(member.name).name
            dest = image_dir / (Path(name).stem + ".jpg")
            if dest.exists():
                count += 1
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            tmp = image_dir / f".tmp_{name}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(extracted.read())
            try:
                _save_resized(tmp, dest, image_size=image_size)
                count += 1
            finally:
                tmp.unlink(missing_ok=True)
    if not keep_tar:
        tar_path.unlink(missing_ok=True)
    return count


def download_images(
    raw_dir: Path,
    image_dir: Path,
    image_size: int = 224,
    max_tars: int | None = None,
    start_index: int = 1,
    keep_tar: bool = False,
) -> int:
    """Download NIH image archives. `max_tars=1` is enough to start training."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    selected = NIH_IMAGE_TARS[start_index - 1 :]
    if max_tars is not None:
        selected = selected[:max_tars]
    for url, filename in selected:
        tar_path = raw_dir / filename
        if not tar_path.exists():
            print(f"Downloading {filename} ...")
            _http_get(url, tar_path, timeout=600)
        total += process_tar(tar_path, image_dir, image_size=image_size, keep_tar=keep_tar)
        print(f"Processed {filename}: running image count ≈ {total}")
    return total
