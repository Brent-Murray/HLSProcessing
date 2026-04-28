import re
import shutil
from datetime import datetime
from pathlib import Path

import rasterio
from rasterio.merge import merge
from tqdm import tqdm


def extract_metadata(path):
    name = path.name

    date_pattern = re.compile(r"\.(\d{7})T\d{6}\.")  # 2025182
    tile_pattern = re.compile(r"\.(T\d{2}[A-Z]{3})\.")  # T07VDG
    band_pattern = re.compile(r"\.(B\d{2}|B\dA|Fmask|SZA|SAA|VZA|VAA)\.tif$")

    date_match = date_pattern.search(name)
    tile_match = tile_pattern.search(name)
    band_match = band_pattern.search(name)

    if not all([date_match, tile_match, band_match]):
        return None

    julian_date = date_match.group(1)  # e.g. 2025182
    date = datetime.strptime(julian_date, "%Y%j").strftime("%Y-%m-%d")

    return {
        "date": date,
        "julian_date": julian_date,
        "tile": tile_match.group(1),
        "band": band_match.group(1),
        "path": path,
    }


def extract_tile_date(path):
    name = path.name
    date_pattern = re.compile(r"\.(\d{7})T\d{6}\.")  # 2025182
    tile_pattern = re.compile(r"\.(T\d{2}[A-Z]{3})\.")  # T07VDG

    date_match = date_pattern.search(name)
    tile_match = tile_pattern.search(name)

    if not all([date_match, tile_match]):
        return None

    julian_date = date_match.group(1)

    return {
        "date": datetime.strptime(julian_date, "%Y%j").strftime("%Y-%m-%d"),
        "julian_date": julian_date,
        "tile": tile_match.group(1),
        "path": path,
    }


def create_raster_stacks(
    root_dir,
    out_dir,
    bands_to_include,
    file_pattern="*.tif",
    require_all_bands=True,
):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir) / "composites"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(root_dir.rglob(file_pattern))
    records = []

    for path in tqdm(all_files, desc="Scanning files"):
        meta = extract_metadata(path)
        if meta is None:
            continue

        if meta["band"] in bands_to_include:
            records.append(meta)

    groups = {}
    for rec in records:
        key = (rec["tile"], rec["date"], rec["julian_date"])
        groups.setdefault(key, {})[rec["band"]] = rec["path"]

    print(f"Matched records: {len(records)}")
    print(f"Tile/date groups: {len(groups)}")
    print(f"Found selected bands: {sorted(set(rec['band'] for rec in records))}")

    written = 0
    skipped = 0

    for (tile, date, julian_date), band_files in tqdm(
        groups.items(), desc="Creating stacks"
    ):
        missing = [b for b in bands_to_include if b not in band_files]

        if missing and require_all_bands:
            skipped += 1
            tqdm.write(f"Skipping {tile} {julian_date}: missing {missing}")
            continue
        try:

            available_bands = [b for b in bands_to_include if b in band_files]
            ordered_paths = [band_files[b] for b in available_bands]

            arrays = []
            profile = None

            for path in ordered_paths:
                with rasterio.open(path) as src:
                    if profile is None:
                        profile = src.profile.copy()
                    arrays.append(src.read(1))

            profile.update(
                count=len(arrays),
                driver="GTiff",
                compress="lzw",
            )

            out_path = out_dir / f"{tile}_{julian_date}_stack.tif"

            with rasterio.open(out_path, "w", **profile) as dst:
                for i, arr in enumerate(arrays, start=1):
                    dst.write(arr, i)
                    dst.set_band_description(i, available_bands[i - 1])

            written += 1
        except Exception as e:
            skipped += 1
            tqdm.write(f"Error processing {tile} {julian_date}: {e}")

    print(f"Finished. Wrote {written} stacks. Skipped {skipped} stacks.")


def copy_fmask_files(
    root_dir,
    out_dir,
    file_pattern="*.tif",
):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir) / "fmask"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(root_dir.rglob(file_pattern))

    copied = 0
    skipped = 0

    for path in tqdm(all_files, desc="Copying Fmask files"):
        if "Fmask" not in path.name:
            continue

        meta = extract_metadata(path)
        if meta is None:
            continue

        tile = meta["tile"]
        date = meta["julian_date"]

        dest = out_dir / f"{tile}_{date}_fmask.tif"

        if dest.exists():
            skipped += 1
            continue

        shutil.copy2(path, dest)
        copied += 1

    print(f"Finished. Copied {copied} files. Skipped {skipped} existing files.")


def copy_metadata_json(root_dir, out_dir):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir) / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(root_dir.rglob("*.json"))

    copied = 0
    skipped = 0
    failed_parse = 0

    for path in tqdm(all_files, desc="Copying JSON metadata"):
        meta = extract_tile_date(path)

        if meta is None:
            failed_parse += 1
            continue

        tile = meta["tile"]
        date = meta["julian_date"]

        dest = out_dir / f"{tile}_{date}_metadata.json"

        if dest.exists():
            skipped += 1
            continue

        shutil.copy2(path, dest)
        copied += 1

    print(
        f"Finished. Found {len(all_files)} JSON files. "
        f"Copied {copied}. Skipped {skipped}. Failed to parse {failed_parse}."
    )


if __name__ == "__main__":
    root_dir = Path(r"F:\yukon\hls\S30")
    out_dir = Path(r"F:\yukon\processed\S30")
    out_dir.mkdir(parents=True, exist_ok=True)

    bands_to_include = [  # S30
        "B02",  # blue
        "B03",  # green
        "B04",  # red
        "B05",  # red-edge
        "B06",  # red-edge
        "B07",  # red-edge
        "B08",  # nir
        "B11",  # swir
        "B12",  # swir
    ]

    # bands_to_include = [  # L30
    #     "B02",  # blue
    #     "B03",  # green
    #     "B04",  # red
    #     "B05",  # nir
    #     "B06",  # swir
    #     "B07",  # swir
    # ]

    copy_fmask_files(root_dir, out_dir)
    copy_metadata_json(root_dir, out_dir)
    create_raster_stacks(root_dir, out_dir, bands_to_include)
